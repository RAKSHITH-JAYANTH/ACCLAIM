import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
import math

import intel_extension_for_pytorch
import gc
import os

def sum_all_diagonal_matrix(mat: torch.tensor):

    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), -1) # pads the matrix on left and right
    #mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)) # Change the strides
    mat_strided = mat_padded.as_strided(
        (b, h, n, n + m),  # Preserve the batch and head dimensions
        (h * n * (2 * n + m), n * (2 * n + m), 2 * n + m + 1, 1)  # Strides updated for batch and head dimensions
    )
    sum_diags = torch.sum(mat_strided, 2) # Sums the resulting matrix's columns
    return sum_diags[:,:,1:]

def create_sparse_causal_mask_all_heads(seq_len, vertical_topk, slash_indices, device):
    
    print("entered create function")
    batch_size, num_heads,_, vertical_size = vertical_topk.shape
    slash_size = slash_indices.size(-1)
    vertical_topk = vertical_topk.squeeze(dim=2)

    # Step 1: Combine vertical and slash indices
    combined_keys = torch.cat([vertical_topk, slash_indices], dim=-1)  # Shape: [B, H, vertical_size + slash_size]
    combined_keys = combined_keys.to(device)
    
    # Ensure unique keys per head, sorted and valid
    sparse_keys = torch.sort(combined_keys, dim=-1)[0]  # Sort keys
    
    sparse_keys = sparse_keys[(sparse_keys >= 0) & (sparse_keys < seq_len)].view(batch_size, num_heads, -1)
    num_sparse_keys = sparse_keys.size(-1)  # Unique sparse keys per head

    sparse_mask = None
    

    chunk_size = 500  # Process in chunks to fit memory

    # Precompute the expanded sparse keys to avoid recomputation in the loop
    sparse_keys_expanded = sparse_keys.unsqueeze(2).expand(batch_size, num_heads, seq_len, num_sparse_keys)

    # Compute the sparse mask in chunks
    seq_range = torch.arange(seq_len, device=sparse_keys.device).view(1, 1, -1, 1)

    sparse_keys_expanded = sparse_keys.unsqueeze(2).expand(batch_size, num_heads, seq_len, num_sparse_keys)
    seq_range_expanded = seq_range.expand(batch_size, num_heads, seq_len, num_sparse_keys)
    
    return sparse_mask, sparse_keys

def vs_as_par(prompt_len, x, cache_k, head_dim, xq, device):

    seqlen = prompt_len       
    idx = min(64, seqlen)
    bsz = 1
    q = xq[:,:,:idx, :].to(device)
    q = q.transpose(1,2)
    scores = None
    chunk_size = 8000
    for i in range(0, prompt_len, chunk_size):
        start = i
        end = min(i + chunk_size, prompt_len)
        keys = cache_k[:bsz, start : end]
        keys = torch.tensor(keys).to(device)
        if keys.shape[0] != 1:
            keys = keys.unsqueeze(0)
        
        keys = keys.transpose(1, 2)
        #print(keys.shape)
        #print(xq.shape)
        if scores is None:
            scores = (torch.matmul(q.to(device), keys.transpose(2, 3)) / math.sqrt(head_dim)).to(device)
        else:
            new_scores = (torch.matmul(q.to(device), keys.transpose(2, 3)) / math.sqrt(head_dim)).to(device)
            scores = torch.cat([scores, new_scores], dim=-1)
            del new_scores
        del keys            
        torch.xpu.empty_cache()
    
    #print(scores.shape)
    
    arange = torch.arange(idx, device=device)
    idx_MASK = arange[None, None, :, None] >= arange[None, None, None, :]
    #print(scores[..., -idx:].shape, idx_MASK.shape)
    
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    
    
    torch.xpu.empty_cache()
    vertical = scores.sum(-2, keepdim=True)
    vertical_topk = torch.zeros((1, 1, 1000), dtype=torch.int32)
    slash = torch.zeros((1, 1, 6096), dtype=torch.int32) 
    vertical_size, slash_size  = min(prompt_len, max(300, 30)), min(prompt_len, max(800, 50))
    vertical_topk = vertical_topk[:, :, :vertical_size]
    vertical_topk = (torch.topk(vertical, vertical_size, -1).indices)
    vertical_topk = vertical_topk.to("cpu")

    slash = slash[:, :, :slash_size]
    slash = sum_all_diagonal_matrix(scores)[...,:-idx + 1]
    del scores
    del idx_MASK
    del arange
    del q
    torch.xpu.empty_cache()
    slash = (prompt_len - 1) - torch.topk(slash, slash_size, -1).indices
    slash = slash.to("cpu")
    print("slash and vertical indices ready")
    sparse_mask, sparse_keys = create_sparse_causal_mask_all_heads(prompt_len, vertical_topk, slash, "cpu")

def bs_par(xq, head_idx, mask_bs, score_bs, heads_per_kv, n_local_heads, n_local_kv_heads, head_dim , prompt_len, cache_k, device):
    
    heads_per_kv = n_local_heads //n_local_kv_heads
    
    q = xq[:,head_idx:head_idx+1,:,:].to(device)
    
    block_size = 64
    q_len = prompt_len
    block_num = (q_len - 1) // block_size + 1
    block_q = torch.zeros(1,1,block_num * block_size, head_dim).to(xq).to(device)
    
    block_q[:,:,:q_len] = q.to(device)
    
    block_q = block_q.reshape(1,1,block_num,block_size,-1).mean(-2)
    bsz = 1
    batch_indices = np.arange(bsz)[:, None, None] 
    scores = None
    chunk_size = 2000
    
    for i in range(0, prompt_len, chunk_size):
        
        start = i
        end = min(i + chunk_size, prompt_len)
        keys = cache_k[:bsz, np.arange(start, end)[None, :, None], np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],:]
        keys = torch.tensor(keys).to(device)
        keys = keys.squeeze(-2)
        
        if keys.shape[0] != 1:
            keys = keys.unsqueeze(0)
        block_num_k = ((end-start) - 1) // block_size + 1
        
        block_k = torch.zeros(1,1,block_num_k * block_size, head_dim).to(keys).to(device)
        
        block_k[:,:,:(end-start)] = keys
        block_k = block_k.reshape(1,1,block_num_k,block_size,-1).mean(-2)
        
        scores = (torch.matmul(block_q, block_k.transpose(2, 3)) / math.sqrt(head_dim)).to(device)
        
        prev_size = (i - 1)// block_size + 1                
        
        score_bs[0:block_q.shape[-2], prev_size:prev_size + block_k.shape[-2]] = scores.detach().cpu().numpy() + mask_bs[0:block_q.shape[-2], prev_size:prev_size+block_k.shape[-2]]  
        
        del keys, block_k, scores            
        torch.xpu.empty_cache()
        
    
    topk_values, topk_indices = torch.topk(-torch.tensor(score_bs[0:block_num, 0:block_num]), 100, dim=-1)
    
    block_rows = torch.arange(block_num, device="cpu").view(-1, 1).repeat(1, block_num)
    block_cols = torch.arange(block_num, device="cpu").view(1, -1).repeat(block_num, 1)
    
    block_mask = torch.ones((block_num, block_num), dtype=torch.int32, device="cpu")
    block_mask[torch.arange(block_num, device="cpu").view(-1, 1), topk_indices] = 0
    
    mask_est = block_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, block_size, block_size)
    
    mask_est.copy_(mask_est[:q_len, :q_len])
    
    del block_mask, block_rows, block_cols, block_q,topk_indices, topk_values, q
    torch.xpu.empty_cache()
    
    
    mask_est.copy_(torch.tril(mask_est))
    mask_est.copy_(mask_est == 0)
    
    mask_flat = mask_est.flatten()
    mask_est2_size = mask_est.size(1)
    del mask_est
    torch.xpu.empty_cache()

    nonzero_indices1 = (torch.where(mask_flat == False)[0])
    nonzero_indices = nonzero_indices1
    del mask_flat
    torch.xpu.empty_cache()
    
    rows = (nonzero_indices // mask_est2_size)
    
    row_counts = torch.bincount(rows)
    unique_rows = torch.nonzero(row_counts).squeeze(1)

    del rows, row_counts
    torch.xpu.empty_cache()

    cols = nonzero_indices % mask_est2_size
    
    col_counts = torch.bincount(cols)
    unique_cols = torch.nonzero(col_counts).squeeze(1)
    
    del cols, col_counts
    torch.xpu.empty_cache()
    
    del nonzero_indices
    torch.xpu.empty_cache()
        
    del unique_cols, unique_rows
    gc.collect()
    torch.xpu.empty_cache()

def profile_ops(n_kv_heads, n_local_heads, n_local_kv_heads, n_rep, head_dim, device_list, bs_heads, avs_heads):

    avs_perf = {}
    bs_perf = {}
    for device in device_list:
        avs_perf[device] = 0
        bs_perf[device] = 0

        for prompt_len in [8000, 16000, 32000, 64000]:
            cache_k = np.memmap(
                rf"C:\Users\raksh\Documents\Accel-Long\cache_k_profile.dat",
                dtype="float32",
                mode="w+",
                shape=(
                    1,
                    prompt_len,
                    1,
                    head_dim,
                ),
            )        
            cache_k[:] = 1.0
            x = torch.randn(1, prompt_len, head_dim)
            xq = torch.randn(1, prompt_len, 1, head_dim)
            start = time.time()
            vs_as_par(prompt_len, x, cache_k, head_dim, xq, device)
            end = time.time()

            latency = end - start
            avs_perf[device] += latency

            mask_bs = np.memmap(
                rf"C:\Users\raksh\Documents\Accel-Long\mask_profile.dat",
                dtype="int32",
                mode="w+",
                shape=(
                    prompt_len,
                    prompt_len,
                ),
            )

            score_bs = np.memmap(
                rf"C:\Users\raksh\Documents\Accel-Long\score_profile.dat",
                dtype="float32",
                mode="w+",
                shape=(
                    prompt_len,
                    prompt_len,
                ),
            )

            start = time.time()
            bs_par(xq, 0, mask_bs, score_bs, 1,1,1, head_dim , prompt_len, cache_k, device)
            end = time.time()

            latency = end - start
            bs_perf[device] += latency

            del cache_k, mask_bs, score_bs
            os.remove(rf"C:\Users\raksh\Documents\Accel-Long\cache_k_profile.dat")
            os.remove(rf"C:\Users\raksh\Documents\Accel-Long\mask_profile.dat")
            os.remove(rf"C:\Users\raksh\Documents\Accel-Long\score_profile.dat")
        
        avs_perf[device] /= 4
        bs_perf[device] /= 4
    
        print("Device: ", device)
        print("AVS: ", avs_perf[device], " sec")
        print("BS: ", bs_perf[device], " sec")

profile_ops(1,1,1, 1, 3072 , [ "xpu"], 2, 30)           

