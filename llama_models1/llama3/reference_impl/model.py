# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch import nn
import numpy as np

from ..api import ModelArgs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent
import gc
import threading
import copy

import time
import os
import json

from collections import Counter


# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # print("freqs_cis.shape", freqs_cis.shape)
    # print("x.shape[1], x.shape[-1]", x.shape[1], x.shape[-1])
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)



def apply_rotary_emb_k(
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor]:
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xk_)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xk_out.type_as(xk)

def apply_rotary_emb_q(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

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
    

    print("completed mask init")

    chunk_size = 500  # Process in chunks to fit memory

    # Precompute the expanded sparse keys to avoid recomputation in the loop
    sparse_keys_expanded = sparse_keys.unsqueeze(2).expand(batch_size, num_heads, seq_len, num_sparse_keys)

    # Compute the sparse mask in chunks
    seq_range = torch.arange(seq_len, device=sparse_keys.device).view(1, 1, -1, 1)

    sparse_keys_expanded = sparse_keys.unsqueeze(2).expand(batch_size, num_heads, seq_len, num_sparse_keys)
    seq_range_expanded = seq_range.expand(batch_size, num_heads, seq_len, num_sparse_keys)
        
    print("final loop done")
    return sparse_mask, sparse_keys





# WRITES TO DISK : STARTS HERE
class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False).to("xpu")
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False).to("xpu")
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False).to("xpu")
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False).to("xpu")
        

        # Paths to memory-mapped files
        self.cache_k_path = rf".\cache_k1_{layer_id}.dat"
        self.cache_v_path = rf".\cache_v1_{layer_id}.dat"
        self.mask_bs_path = rf".\mask_bs_{layer_id}.dat"
        self.score_bs_path = rf".\score_bs_{layer_id}.dat"
        self.mask_est_path = rf".\mask_est_{layer_id}.dat"
        if os.path.exists(self.cache_k_path):
            os.remove(self.cache_k_path)
        
        if os.path.exists(self.cache_v_path):
            os.remove(self.cache_v_path)

        # if os.path.exists(self.mask_bs_path):
        #     os.remove(self.mask_bs_path)

        # if os.path.exists(self.score_bs_path):
        #     os.remove(self.score_bs_path)

        if os.path.exists(self.mask_est_path):
            os.remove(self.mask_est_path)

        self.max_seq_len = args.max_seq_len
        # Initialize memory-mapped files for cache_k and cache_v
        self.cache_k = np.memmap(
            self.cache_k_path,
            dtype="float32",
            mode="w+",
            shape=(
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
        )
        self.cache_v = np.memmap(
            self.cache_v_path,
            dtype="float32",
            mode="w+",
            shape=(
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
        )

        self.cache_k[:] = 0.0
        self.cache_v[:] = 0.0

        self.vertical_topk = torch.zeros((args.max_batch_size, self.n_local_heads, 1000), dtype=torch.int32) #1000
        self.slash = torch.zeros((args.max_batch_size, self.n_local_heads, 6096), dtype=torch.int32) #6096
        self.sparse_mask = None
        self.sparse_keys = None
        self.print1 = 0

        self.bs_idx = {i: [] for i in range(self.n_local_heads)}


        with open("distribution.json", "r") as f:
            self.main_dict = json.load(f)
        
        self.distribution_dict = self.main_dict[str(layer_id)]

       
    
    def kv_cache_update(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):

        x = torch.tensor(x)
        bsz, seqlen, _ = x.shape

        # Split data for CPU and XPU computation
        split_dim = x.shape[1] // x.shape[1] # Split 1/3rd on CPU, 2/3rd on XPU
        x_cpu, x_xpu = x[:, :split_dim, :], x[:, split_dim:, :]

        # Transfer data to respective devices
        x_cpu = x_cpu.to("cpu").contiguous()
        x_xpu = x_xpu.to("xpu").contiguous()
        freqs_cis_xpu = freqs_cis.to("xpu")

        # Efficiently duplicate self.wk and self.wv for CPU computation
        wk_cpu = copy.deepcopy(self.wk).to("cpu")
        wv_cpu = copy.deepcopy(self.wv).to("cpu")

        # Define async tasks for CPU and XPU computation
        def compute_on_cpu():
            nonlocal xk_cpu, xv_cpu
            xk_cpu, xv_cpu = wk_cpu(x_cpu), wv_cpu(x_cpu)

        def compute_on_xpu():
            nonlocal xk_xpu, xv_xpu
            xk_xpu, xv_xpu = self.wk(x_xpu), self.wv(x_xpu)

        # Thread creation
        xk_cpu, xv_cpu = None, None
        xk_xpu, xv_xpu = None, None
        thread_cpu = threading.Thread(target=compute_on_cpu)
        thread_xpu = threading.Thread(target=compute_on_xpu)

        # Start and join threads
        thread_cpu.start()
        thread_xpu.start()
        thread_cpu.join()
        thread_xpu.join()

        # Combine results before reshaping
        xk_combined = torch.cat([xk_cpu.to("cpu"), xk_xpu.to("cpu")], dim=1)
        xv_combined = torch.cat([xv_cpu.to("cpu"), xv_xpu.to("cpu")], dim=1)

        # Reshape after concatenation
        xk_combined = xk_combined.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv_combined = xv_combined.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xk_combined = apply_rotary_emb_k(xk_combined, freqs_cis=freqs_cis.to("cpu"))

        # Cache update
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk_combined.detach().cpu().numpy()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv_combined.detach().cpu().numpy()

        # Clean up
        del x, xk_cpu, xv_cpu, xk_xpu, xv_xpu, xk_combined, xv_combined, wk_cpu, wv_cpu
        torch.xpu.empty_cache()

    
    def mask_generation(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        chunk_size: int,
        prompt_len: int,
        mask_bs,
        score_bs,
    ):
        
        print("enter mask creation")
        start = time.time()
        x = torch.tensor(x)
        bsz, seqlen, _ = x.shape
        x = x.to("xpu")
        bsz, seqlen, _ = x.shape
        xq = self.wq(x)
        del x
        torch.xpu.empty_cache()
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq1 = apply_rotary_emb_q(xq, freqs_cis=freqs_cis.to("xpu"))
        del xq
        torch.xpu.empty_cache()
        xq = xq1
        xq = xq.transpose(1, 2)
        head_type = "BS"
        
        def vs_as_par(xq, bsz = bsz, seqlen = seqlen):
            
            
            idx = min(64, seqlen)
            
            q = xq[:,:,:idx, :]
            scores = None
            chunk_size = 8000
            for i in range(0, prompt_len, chunk_size):
                start = i
                end = min(i + chunk_size, prompt_len)
                keys = self.cache_k[:bsz, start : end]
                keys = torch.tensor(keys).to("xpu")
                if keys.shape[0] != 1:
                    keys = keys.unsqueeze(0)
                keys = repeat_kv(
                    keys, self.n_rep
                )
                keys = keys.transpose(1, 2)
                if scores is None:
                    scores = (torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim))
                else:
                    new_scores = (torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim))
                    scores1 = torch.cat([scores, new_scores], dim=-1)
                    
                    del new_scores, scores
                    scores = scores1
                del keys            
                torch.xpu.empty_cache()

            arange = torch.arange(idx, device="xpu")
            idx_MASK = arange[None, None, :, None] >= arange[None, None, None, :]
            scores[:, :, :, -idx:] = torch.where(idx_MASK[...,-idx:,-idx:].to("xpu"), scores[:, :, :, -idx:], -torch.inf)
            scores1 = F.softmax(scores.float(), dim=-1).type_as(xq)
            del scores           
            torch.xpu.empty_cache()
            scores = scores1
            vertical = scores.sum(-2, keepdim=True)
            vertical_size, slash_size  = min(prompt_len, max(300, 30)), min(prompt_len, max(800, 50)) #  CHANGE HERE AS WELL
            self.vertical_topk = self.vertical_topk[:, :, :vertical_size]
            self.vertical_topk = (torch.topk(vertical, vertical_size, -1).indices)
            self.vertical_topk = self.vertical_topk.to("cpu")

            self.slash = self.slash[:, :, :slash_size]
            self.slash = sum_all_diagonal_matrix(scores)[...,:-idx + 1]
            del scores
            del idx_MASK
            del arange
            del q
            del vertical
            torch.xpu.empty_cache()
            self.slash = (prompt_len - 1) - torch.topk(self.slash, slash_size, -1).indices
            self.slash = self.slash.to("cpu")
            #print("slash and vertical indices ready")
            self.sparse_mask, self.sparse_keys = create_sparse_causal_mask_all_heads(prompt_len, self.vertical_topk, self.slash, "cpu")
        
        def bs_par(xq, head_idx, mask_bs, score_bs):
            
            #print("Hi Enter BS")
            heads_per_kv = self.n_local_heads //self.n_local_kv_heads
            
            q = xq[:,head_idx:head_idx+1,:,:]
            
            block_size = 64
            q_len = seqlen
            block_num = (q_len - 1) // block_size + 1
            block_q = torch.zeros(1,1,block_num * block_size, self.head_dim).to(xq)
            
            block_q[:,:,:q_len] = q
            
            block_q = block_q.reshape(1,1,block_num,block_size,-1).mean(-2)
            
            batch_indices = np.arange(bsz)[:, None, None] 
            scores = None
            chunk_size = 4000
            
            for i in range(0, prompt_len, chunk_size):
                
                start = i
                end = min(i + chunk_size, prompt_len)
                keys = self.cache_k[:bsz, np.arange(start, end)[None, :, None], np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],:]
                keys = torch.tensor(keys).to("xpu")
                keys = keys.squeeze(-2)
                
                if keys.shape[0] != 1:
                    keys = keys.unsqueeze(0)
                block_num_k = ((end-start) - 1) // block_size + 1
                
                block_k = torch.zeros(1,1,block_num_k * block_size, self.head_dim).to(keys)
                
                block_k[:,:,:(end-start)] = keys
                block_k = block_k.reshape(1,1,block_num_k,block_size,-1).mean(-2)
                
                scores = (torch.matmul(block_q, block_k.transpose(2, 3)) / math.sqrt(self.head_dim)).to("xpu")
                
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
            
            #print("Mask size: ", mask_est.shape)
            mask_est = mask_est.view(self.max_seq_len, self.max_seq_len)
            #mask_est.copy_(torch.tril(mask_est))
            idx = torch.arange(self.max_seq_len)
            mask =  idx[:, None] >= idx[None, :]
            #print("Mask.shape: ", mask.shape)
            
            mask_est &= mask
            del mask
            #print("Tril done")
            #mask_est.copy_(mask_est == 0)

            
            mask_est = mask_est[:q_len, :q_len]
            #print("mask shape: ", mask_est.shape)

            row_mask = mask_est.all(dim=1)
            unique_rows = torch.nonzero(row_mask).squeeze(1)

            #print("Row done")

            col_mask = mask_est.all(dim=0)
            unique_cols = torch.nonzero(col_mask).squeeze(1)

            #print("col done")

            del mask_est, row_mask, col_mask
            
            # mask_flat = mask_est.flatten()
            # mask_est2_size = mask_est.size(1)
            # del mask_est
            # torch.xpu.empty_cache()
            # print("Hi1")
            # nonzero_indices = (torch.where(mask_flat == False)[0])
            # #nonzero_indices = nonzero_indices1
            # del mask_flat
            # torch.xpu.empty_cache()
            
            # rows = (nonzero_indices // mask_est2_size)
            # print("Hi2")
            # row_counts = torch.bincount(rows)
            # unique_rows = torch.nonzero(row_counts).squeeze(1)
            # max_row = torch.max(rows)
            # print(max_row)
            # print("Hi3")
            # del rows, row_counts
            # torch.xpu.empty_cache()

            # cols = nonzero_indices % mask_est2_size
            
            # col_counts = torch.bincount(cols)
            # unique_cols = torch.nonzero(col_counts).squeeze(1)
            
            # print("Hi4")
            # del cols, col_counts
            # torch.xpu.empty_cache()
            
            # del nonzero_indices
            # torch.xpu.empty_cache()


            self.bs_idx[head_idx].append((unique_rows))
            
            self.bs_idx[head_idx].append((unique_cols))
            
            del unique_cols, unique_rows
            gc.collect()
            torch.xpu.empty_cache()
            
            
        count_bs = 0
        idx_list = []
        for sub_key, value in self.distribution_dict.items():
            if value[0] == "BS":
                count_bs += 1
                idx_list.append(int(sub_key))
        streams = [torch.xpu.Stream() for _ in range(count_bs % 2 + 1)]
        
        with torch.xpu.stream(streams[count_bs % 2 ]):
            vs_as_par(xq)

        # mask_bs = np.memmap(
        #     self.mask_bs_path,
        #     dtype="int32",
        #     mode="w+",
        #     shape=(
        #         self.max_seq_len,
        #         self.max_seq_len,
        #     ),
        # )

        # score_bs = np.memmap(
        #     self.score_bs_path,
        #     dtype="float32",
        #     mode="w+",
        #     shape=(
        #         self.max_seq_len,
        #         self.max_seq_len,
        #     ),
        # )
        # print("Mask init")
        # for i in range(self.max_seq_len):
        #     mask_bs[i, :i + 1] = 1
        # print("Mask complete")
        # # Flush data to ensure it's written to disk
        # mask_bs.flush()
        # print("Mask Flush")

        for idx in idx_list:
            with torch.xpu.stream(streams[idx % 1]):
                bs_par(xq, idx, mask_bs, score_bs)
        
        torch.xpu.synchronize()


        del xq #, mask_bs, score_bs

        end = time.time()
        torch.xpu.empty_cache()
        gc.collect()

        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) and obj.device.type == 'xpu':
        #             print(f"Unfreed XPU tensor: {obj.size()}")
        #     except Exception:
        #         pass

        print("mask creation complete")
        print("Mask Time = ", end - start)
        

    def prefill_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        chunk_size: int,
        prompt_len: int,
    ):
        
        bsz, seqlen, _ = x.shape
        
        if torch.isnan(x).any():
            print("NaN detected in x!")
        
        xq = self.wq(x)
        del x
        torch.xpu.empty_cache()
        if torch.isnan(xq).any():
            print("NaN detected in xq!")

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = apply_rotary_emb_q(xq, freqs_cis=freqs_cis.to("xpu"))

        xq = xq.transpose(1, 2) #.to("xpu")
           
        chunked_sparse_keys = self.sparse_keys.to("cpu").split(chunk_size)

        selective_indices = self.sparse_keys
        
        if torch.isnan(selective_indices).any():
            print("NaN detected in selective_indices!")

        bs = bsz

        heads_per_kv = self.n_local_heads //self.n_local_kv_heads

        scores = {i: None for i in range(self.n_local_heads)}
        
        num_sparse_keys = self.sparse_keys.size(-1)
        mask = None

        mask = torch.zeros((bsz, self.n_local_heads, seqlen, num_sparse_keys))
        
        seq_range = torch.arange(start_pos, start_pos + seqlen, device="xpu").view(1, 1, -1, 1)
        sparse_keys_expanded = self.sparse_keys.unsqueeze(2).expand(bsz, self.n_local_heads, seqlen, num_sparse_keys).to("xpu")
        seq_range_expanded = seq_range.expand(bsz, self.n_local_heads, seqlen, num_sparse_keys).to("xpu")

        mask[:, :, :, :] = torch.where(
            sparse_keys_expanded[:, :, :, :] > seq_range_expanded[:, :, :, :],
            -1e9,
            0.0
        )
        mask = mask.to("xpu")
        del seq_range
        del sparse_keys_expanded
        del seq_range_expanded
        torch.xpu.empty_cache()

        
        if torch.isnan(mask).any():
            print("NaN detected in mask!")
        
        

        output = {i: None for i in range(self.n_local_heads)}

        def par_heads_cpu(head_idx, mask, device):
            for chunk in chunked_sparse_keys:
            
                selected_indices = chunk
            
                selective_indices_reduced = selective_indices[:, ::heads_per_kv, :]  # Shape: [bsz, 8, 18]
                k_indices = selective_indices[:, head_idx:head_idx+1, :]

                batch_indices = np.arange(bsz)[:, None, None]  # Shape: [bsz, 1, 1]
                kv_head_indices = np.arange(head_idx)[None, :, None]  # Shape: [1, n_kv_heads, 1]
                gathered_keys = self.cache_k[
                    batch_indices,  # Batch dimension
                    k_indices,  # Selected sequence indices (num_keys per KV head)
                    np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],  # KV head indices
                    :  # All dimensions
                ]# Shape: [bsz, 8, 18, 128]
                
                gathered_keys = gathered_keys.transpose(0, 2, 1, 3)
                
                keys = torch.tensor(gathered_keys).to(device)  # Convert to PyTorch tensor
                
                keys = keys.transpose(1, 2)

                if torch.isnan(keys).any():
                    print("NaN detected in keys!", chunk)
                if scores[head_idx] == None:
                    scores[head_idx] = (torch.matmul(xq[:,head_idx:head_idx+1,:,:].to(device), keys.transpose(2, 3)) / math.sqrt(self.head_dim)).to(device)
                    
                else:
                    new_scores = torch.matmul(xq[:,head_idx:head_idx+1,:,:].to(device), keys.transpose(2, 3)) / math.sqrt(self.head_dim)                    
                    scores[head_idx] = torch.cat([scores[head_idx], new_scores], dim=-1)
                    del new_scores
                del keys
                torch.xpu.empty_cache()

                if torch.isnan(scores[head_idx]).any():
                    print("NaN detected in scores!")
                    
            
            mask = mask.to(device)
            if mask is not None:
                if torch.isnan(scores[head_idx]).any():
                    print("NaN detected in scores!")
                
                if torch.isnan(mask).any():
                    print("NaN detected in mask!")
                scores[head_idx] = scores[head_idx] + mask

                if torch.isnan(scores[head_idx]).any():
                    print("NaN detected in scores-2!")
            
            scores[head_idx] = F.softmax(scores[head_idx].float(), dim=-1)
        
                    
            if torch.isnan(scores[head_idx]).any():
                print("NaN detected in scores-3!")

        
            start_idx = 0

            for chunk in chunked_sparse_keys:
                end_idx = start_idx + chunk.size(-1)

                selected_indices = torch.tensor(chunk)
            
                selective_indices_reduced = selective_indices[:, ::heads_per_kv, :]
                v_indices = selective_indices[:, head_idx:head_idx+1, :]
                
                batch_indices = np.arange(bsz)[:, None, None]  # Shape: [bsz, 1, 1]
                kv_head_indices = np.arange(self.n_local_kv_heads)[None, :, None]  # Shape: [1, n_kv_heads, 1]

                gathered_values = self.cache_v[
                    batch_indices,  # Batch dimension
                    v_indices,  # Selected sequence indices (num_keys per KV head)
                    np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],  # KV head indices
                    :  # All dimensions
                ]# Shape: [bsz, 8, 18, 128]
            
                gathered_values = gathered_values.transpose(0, 2, 1, 3)

                values = torch.tensor(gathered_values).to(device)
                
                values = values.transpose(
                    1, 2
                )
                if torch.isnan(values).any():
                    print("NaN detected in values!", chunk)

                if output[head_idx] is None:
                    output[head_idx] = (torch.matmul(scores[head_idx][:, :, :, start_idx:end_idx], values)).to(device)
                else:
                    output[head_idx] += (torch.matmul(scores[head_idx][:, :, :, start_idx:end_idx], values)).to(device)
                
                if torch.isnan(output[head_idx]).any():
                    print("NaN detected in output!", chunk)
                
                del values
                torch.xpu.empty_cache()
                output[head_idx] = output[head_idx].to("xpu")
                
                start_idx = end_idx
        
        def par_heads_gpu(head_idx, mask, device):
            start_vs = time.time()
            for chunk in chunked_sparse_keys:
            
                selected_indices = chunk
            
                selective_indices_reduced = selective_indices[:, ::heads_per_kv, :]  # Shape: [bsz, 8, 18]
                k_indices = selective_indices[:, head_idx:head_idx+1, :]
                
                batch_indices = np.arange(bsz)[:, None, None]  # Shape: [bsz, 1, 1]
                kv_head_indices = np.arange(head_idx)[None, :, None]  # Shape: [1, n_kv_heads, 1]
                gathered_keys = self.cache_k[
                    batch_indices,  # Batch dimension
                    k_indices,  # Selected sequence indices (num_keys per KV head)
                    np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],
                    :  # All dimensions
                ]# Shape: [bsz, 8, 18, 128]
                
                gathered_keys = gathered_keys.transpose(0, 2, 1, 3)
                
                keys = torch.tensor(gathered_keys).to(device)
                
                keys = keys.transpose(1, 2)

                if torch.isnan(keys).any():
                    print("NaN detected in keys!", chunk)

                if scores[head_idx] == None:
                    scores[head_idx] = (torch.matmul(xq[:,head_idx:head_idx+1,:,:].to(device), keys.transpose(2, 3)) / math.sqrt(self.head_dim)).to(device)
                    
                else:
                    new_scores = torch.matmul(xq[:,head_idx:head_idx+1,:,:].to(device), keys.transpose(2, 3)) / math.sqrt(self.head_dim)                    
                    scores[head_idx] = torch.cat([scores[head_idx], new_scores], dim=-1)
                    
                    del new_scores
                del keys
                torch.xpu.empty_cache()

                if torch.isnan(scores[head_idx]).any():
                    print("NaN detected in scores!")
                    
                            
            mask = mask.to(device)
            if mask is not None:
                if torch.isnan(scores[head_idx]).any():
                    print("NaN detected in scores!")
                
                if torch.isnan(mask).any():
                    print("NaN detected in mask!")
                scores[head_idx] = scores[head_idx] + mask

                if torch.isnan(scores[head_idx]).any():
                    print("NaN detected in scores-2!")
            
            scores[head_idx] = F.softmax(scores[head_idx].float(), dim=-1)
        
            if torch.isnan(scores[head_idx]).any():
                print("NaN detected in scores-3!")

        
            start_idx = 0
            
            for chunk in chunked_sparse_keys:
                end_idx = start_idx + chunk.size(-1)

                selected_indices = torch.tensor(chunk)
            
                selective_indices_reduced = selective_indices[:, ::heads_per_kv, :]
                v_indices = selective_indices[:, head_idx:head_idx+1, :]
                
                batch_indices = np.arange(bsz)[:, None, None]  # Shape: [bsz, 1, 1]
                kv_head_indices = np.arange(self.n_local_kv_heads)[None, :, None]  # Shape: [1, n_kv_heads, 1]

                gathered_values = self.cache_v[
                    batch_indices,  # Batch dimension
                    v_indices,  # Selected sequence indices (num_keys per KV head)
                    np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],  # KV head indices
                    :  # All dimensions
                ]# Shape: [bsz, 8, 18, 128]
            
                gathered_values = gathered_values.transpose(0, 2, 1, 3)

                values = torch.tensor(gathered_values).to(device)
                
                values = values.transpose(
                    1, 2
                )
                if torch.isnan(values).any():
                    print("NaN detected in values!", chunk)

                if output[head_idx] is None:
                    output[head_idx] = (torch.matmul(scores[head_idx][:, :, :, start_idx:end_idx], values)).to(device)
                else:
                    output[head_idx] += (torch.matmul(scores[head_idx][:, :, :, start_idx:end_idx], values)).to(device)
                
                if torch.isnan(output[head_idx]).any():
                    print("NaN detected in output!", chunk)
                
                del values
                torch.xpu.empty_cache()
                output[head_idx] = output[head_idx].to("xpu")
                
                start_idx = end_idx
            
            end_vs = time.time()
            #print("Time VS = ", end_vs - start_vs, " Device = ", device)
        

        def par_heads_bs(head_idx, mask, device):
            start_bs = time.time()          
            mask_rows = torch.tensor(self.bs_idx[head_idx][0]).to(device)
            mask_cols = torch.tensor(self.bs_idx[head_idx][1]).to(device)
            
            end = min(start_pos + chunk_size, prompt_len)
            mask_in_range = (mask_rows >= start_pos) & (mask_rows < end)  # Boolean mask for filtering
            filtered_rows = mask_rows[mask_in_range].detach().cpu().numpy()  # Filtered row indices
            filtered_cols = mask_cols.detach().cpu().numpy() #[mask_in_range]  # Corresponding column indices
            
            if  filtered_rows.shape[0] == 0:
                output[head_idx] = torch.full((1, 1, end - start_pos, self.head_dim), float(0.0), device=device)
                return
            batch_indices = np.arange(bsz)[:, None, None]  # Shape: [bsz, 1, 1]
            kv_head_indices = np.arange(head_idx)[None, :, None]  # Shape: [1, n_kv_heads, 1]
            gathered_keys = self.cache_k[
                batch_indices,  # Batch dimension
                filtered_cols,  # Selected sequence indices (num_keys per KV head)
                np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],  # KV head indices
                :  # All dimensions
            ]# Shape: [bsz, 8, 18, 128]
            
            gathered_keys = gathered_keys.transpose(0, 2, 1, 3)
            
            keys = torch.tensor(gathered_keys).to(device)  # Convert to PyTorch tensor
            
            keys = keys.transpose(1, 2)
            keys = keys.transpose(2, 3)
            keys = keys.squeeze(1)
            # print("keys shape ", keys.shape)
            if torch.isnan(keys).any():
                print("NaN detected in keys!", chunk)
            
            xq_sliced = (xq[:,head_idx,:,:]).to(device)
            scores[head_idx] = (torch.bmm(xq_sliced, keys) / math.sqrt(self.head_dim)).to(device)
            
            del keys, xq_sliced, mask_rows, mask_cols
            torch.xpu.empty_cache()
            scores[head_idx] = torch.tensor(scores[head_idx])
            if torch.isnan(scores[head_idx]).any():
                print("NaN detected in scores!")
            
            
            mask = torch.full((end - start_pos, filtered_cols.shape[0]), -1e9, device=device)

            row_indices = torch.arange(start_pos, end).view(-1, 1)

            valid_mask = torch.tensor(filtered_cols[None, :]) <= row_indices
            mask[valid_mask] = 0

            scores[head_idx] += mask          
            scores[head_idx] = F.softmax(scores[head_idx].float(), dim=-1)
            
                
            if torch.isnan(scores[head_idx]).any():
                print("NaN detected in scores-3!")

            batch_indices = np.arange(bsz)[:, None, None]
            gathered_values = self.cache_v[
                batch_indices,  # Batch dimension
                filtered_cols,  # Selected sequence indices (num_keys per KV head)
                np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],  # KV head indices
                :  # All dimensions
            ]# Shape: [bsz, 8, 18, 128]
            gathered_values = gathered_values.transpose(0, 2, 1, 3)

            values = torch.tensor(gathered_values).to(device)
            values = values.transpose(
                1, 2
            )
            if torch.isnan(values).any():
                print("NaN detected in values!", chunk)
            
            final_result = (torch.bmm(scores[head_idx], values[0,:,:,:])).to(device)
            
            output[head_idx] = final_result.unsqueeze(0)
            if torch.isnan(output[head_idx]).any():
                print("NaN detected in output!", head_idx)
            
            end_bs = time.time()

            #print("BS time = ", end_bs - start_bs, " Device = ", device)
            
            del mask
            del final_result
            del values
            del row_indices, valid_mask, filtered_rows, filtered_cols, scores[head_idx]
            torch.xpu.empty_cache()
            gc.collect()

        a = time.time()
                
        streams = [torch.xpu.Stream() for _ in range((32 // 12) + 1)]

        # for sub_key, value in self.distribution_dict.items():
        #     if(value[1] == "xpu"):
        #         with torch.xpu.stream(streams[-1]):
        #             if(value[0] == "VS" or value[0] == "A"):
        #                 par_heads_gpu(int(sub_key), mask[:, int(sub_key):int(sub_key)+1, :, :], value[1]) # THIS IS FOR VS
        #             else:
        #                 par_heads_bs(int(sub_key), None, value[1])
        #     else:
        #         with torch.xpu.stream(streams[int(sub_key) // 12]):
        #             if(value[0] == "VS" or value[0] == "A"):
        #                 par_heads_gpu(int(sub_key), mask[:, int(sub_key):int(sub_key)+1, :, :], value[1]) # THIS IS FOR VS
        #             else:
        #                 par_heads_bs(int(sub_key), None, value[1])

        # torch.xpu.synchronize()
        def run_task(sub_key, value, mask):
            stream = torch.xpu.Stream()
            with torch.xpu.stream(stream):
                if value[0] in ("VS", "A"):
                    par_heads_gpu(int(sub_key), mask[:, int(sub_key):int(sub_key)+1, :, :], value[1])
                else:
                    par_heads_bs(int(sub_key), None, value[1])

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(run_task, sub_key, value, mask)
                for sub_key, value in self.distribution_dict.items()
            ]
        b = time.time()
        #print("func time = ", b - a)
        output1 = torch.cat(list(output.values()), dim=1)
        output2 = output1.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = output2.to("xpu") 
        out = self.wo(output)
        del output
        del output1
        del output2
        del xq
        #del scores
        del streams
        
        torch.xpu.empty_cache()
        gc.collect()

        torch.xpu.empty_cache()
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) and obj.device.type == 'xpu':
        #             print(f"Unfreed XPU tensor: {obj.size()}")
        #     except Exception:
        #         pass
        return out

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        chunk_size: int,
        is_prefill: bool,
        prompt_len: int,
    ):
        
        if is_prefill:
            return self.prefill_forward(x, start_pos, freqs_cis, chunk_size, prompt_len)
        #print("Hi entered decode layer forward")
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = apply_rotary_emb_q(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb_k(xk, freqs_cis=freqs_cis)


        # Write to memory-mapped files
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.cpu().numpy()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.cpu().numpy()
        
        
        # ACCESS TO KV CHUNKS STARTS HERE
        xq = xq.transpose(1, 2)
        
        scores = None
        for i in range(0, start_pos + seqlen, chunk_size):
            start = i
            end = min(i + chunk_size, start_pos + seqlen)
            keys = self.cache_k[:bsz, start : end]
            keys = torch.tensor(keys, device=xq.device)
            
            if keys.shape[0] != 1:
                keys = keys.unsqueeze(0)

            keys = repeat_kv(
                keys, self.n_rep
            )
            keys = keys.transpose(1, 2)
            
            if scores is None:
                scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            else:
                new_scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
                scores = torch.cat([scores, new_scores], dim=-1)
        
                
        if mask is not None:
            scores = scores + mask
        
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        

        output = None
        for i in range(0, start_pos + seqlen, chunk_size):
            start = i
            end = min(i + chunk_size, start_pos + seqlen)
            values = self.cache_v[:bsz, start : end]
            values = torch.tensor(values, device=xq.device)
            if values.shape[0] != 1:
                values = values.unsqueeze(0)
            values = repeat_kv(
                values, self.n_rep
            )
            values = values.transpose(
                1, 2
            )
            
            if output is None:
                output = torch.matmul(scores[:, :, :, start:end], values)
            else:
                output += torch.matmul(scores[:, :, :, start:end], values)
        
        
        self.print1 = 1
        
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1).to("xpu")
        return self.wo(output).to("xpu")

        

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        # import pdb
        # pdb.set_trace()
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


        self.w1 = nn.Linear(dim, hidden_dim, bias=False).to("xpu")
        self.w2 = nn.Linear(hidden_dim, dim, bias=False).to("xpu")
        self.w3 = nn.Linear(dim, hidden_dim, bias=False).to("xpu")

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, layer_id)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        chunk_size: int,
        is_prefill: bool,
        prompt_len: int,
    ):
        
        
        x1 = self.attention_norm(x.to("cpu"))
        y = self.attention(x1.to("xpu"), start_pos, freqs_cis.to("xpu"), mask, chunk_size, is_prefill, prompt_len)
        h = x.to("xpu") + y
        y1 = self.ffn_norm(h.to("cpu"))
        y1 = y1.to("xpu")
        y2 = self.feed_forward(y1)
        out = h + y2
        del h
        del y
        del y1
        del y2
        del x
        torch.xpu.empty_cache()
        
        return out



class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        # import pdb
        # pdb.set_trace()
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Replace with nn.Embedding
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

        self.embed_path = rf".\input_embed.dat"
        self.mask_bs_path = rf".\mask_bs.dat"
        self.score_bs_path = rf".\score_bs.dat"

        self.dim = params.dim
        # Initialize memory-mapped files for cache_k and cache_v
        self.print_1 = 0
        self.max_seq_len = params.max_seq_len

        self.mask_bs = np.memmap(
            self.mask_bs_path,
            dtype="int32",
            mode="w+",
            shape=(
                self.max_seq_len,
                self.max_seq_len,
            ),
        )

        self.score_bs = np.memmap(
            self.score_bs_path,
            dtype="float32",
            mode="w+",
            shape=(
                self.max_seq_len,
                self.max_seq_len,
            ),
        )
        #print("Mask init")
        for i in range(self.max_seq_len):
            self.mask_bs[i, :i + 1] = 1
        #print("Mask complete")
        # Flush data to ensure it's written to disk
        self.mask_bs.flush()
        #print("Mask Flush")
   
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, is_prefill: bool, begin: int, end: int, prompt_len: int, chunk_size: int):
        
        _bsz, seqlen = tokens.shape
        self.freqs_cis = self.freqs_cis

        h = None
        mask = None
        
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        if is_prefill:
            embed = np.memmap(
                self.embed_path,
                dtype="float32",
                mode="w+",
                shape=(
                    1,
                    seqlen,
                    self.dim,
                ),
            )

            
            #print("Initial embeddings:")
            chunk_size = 2000
            orig_chunk = chunk_size
            for chunk in range(0, seqlen,chunk_size):
                h = self.tok_embeddings(tokens[:, chunk:min(chunk + chunk_size, seqlen)])
                
                embed[:, chunk:min(chunk + chunk_size, seqlen),:] = h 
            
            count_2 = 0
            for layer in self.layers:
                
                chunk_size = orig_chunk
                def process_chunk1(chunk):
                    end = min(chunk + chunk_size, seqlen)
                    
                    freqs_cis = self.freqs_cis[chunk:end]
                    
                    h = torch.tensor(embed[:, chunk:end, :])
                    
                    layer.attention.kv_cache_update(layer.attention_norm(torch.tensor(h)), chunk, freqs_cis)
                    
                with ThreadPoolExecutor(max_workers=32) as executor:
                    futures = {executor.submit(process_chunk1, chunk): chunk for chunk in range(0, seqlen, chunk_size)}

                for future in concurrent.futures.as_completed(futures):
                    chunk = futures[future]
                    try:
                        future.result()
                        
                    except Exception as e:
                        print(f"Chunk {chunk} failed with error: {e}")    
                
                
                idx = min(seqlen, seqlen)
                h = embed[:,seqlen-idx:seqlen,:]
                #print("h , embed=", h.shape, embed.shape)
                freqs_cis = self.freqs_cis[seqlen-idx:seqlen]

                layer.attention.mask_generation(layer.attention_norm(torch.tensor(h)), freqs_cis, chunk_size, seqlen, self.mask_bs, self.score_bs)
                
                chunk_size = 2000
                

                # for chunk in range(0, seqlen,chunk_size):
                #     freqs_cis = self.freqs_cis[chunk:min(chunk + chunk_size, seqlen)]
                #     h = embed[:, chunk:min(chunk + chunk_size, seqlen),:]
                #     h = torch.tensor(h)
                #     h = layer(h, chunk, freqs_cis, mask, chunk_size, is_prefill, seqlen)
                #     #print(h)
                #     embed[:, chunk:min(chunk + chunk_size, seqlen),:] = h.detach().cpu().numpy()
                #     del h
                #     torch.xpu.empty_cache()

                def process_chunk(chunk):
                    chunk_end = min(chunk + chunk_size, seqlen)

                    freqs_cis = self.freqs_cis[chunk:chunk_end]
                    h = embed[:, chunk:chunk_end, :]
                    h = torch.tensor(h)

                    h = layer(h, chunk, freqs_cis, mask, chunk_size, is_prefill, seqlen)

                    embed[:, chunk:chunk_end, :] = h.detach().cpu().numpy()

                    del h
                    

                # Launch threads
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    executor.map(process_chunk, range(0, seqlen, chunk_size))

                
                print("Layer ", count_2)
                count_2 += 1
                #break
            h = torch.tensor(embed[:, seqlen-2:seqlen,:])
            del embed
            del self.mask_bs, self.score_bs
            if os.path.exists(self.mask_bs_path):
                os.remove(self.mask_bs_path)

            if os.path.exists(self.score_bs_path):
                os.remove(self.score_bs_path)
            gc.collect()
        
        else:
        
            h = self.tok_embeddings(tokens)
            if(self.print_1 == 0):
                print("Initial embeddings")
                print(h)
            
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=1)
                mask = torch.hstack(
                    [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
                ).type_as(h)
            # count = 0
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask, chunk_size, is_prefill, seqlen)
                
            self.print_1 = 1
        
        h = self.norm(h.to("cpu"))
        output = self.output(h).float()
        return output

    