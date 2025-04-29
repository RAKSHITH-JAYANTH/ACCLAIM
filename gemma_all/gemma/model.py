# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model implementation."""

import json
import gc
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union, Mapping

from gemma import config as gemma_config
from gemma import tokenizer
import intel_extension_for_pytorch as ipex

import numpy as np
import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import copy
import time
import math


class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: gemma_config.GemmaConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        print("In Sampler ")
        print("Hidden States.shape: ", hidden_states.shape)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids, logits


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         rope_scaling_factor:int = 1) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    freqs = freqs/rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
    ):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

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


class GemmaAttention(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
        layer_id: int,
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=config.quant)
        self.o_proj = Linear(
            self.num_heads * self.head_dim, self.hidden_size, quant=config.quant
        )
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

        self.max_seq_len = config.max_position_embeddings

        self.cache_q_path = rf".\cache_q1_{layer_id}.dat"
        self.cache_k_path = rf".\cache_k1_{layer_id}.dat"
        self.cache_v_path = rf".\cache_v1_{layer_id}.dat"
        self.mask_bs_path = rf".\mask_bs_{layer_id}.dat"
        self.score_bs_path = rf".\score_bs_{layer_id}.dat"
        self.mask_est_path = rf".\mask_est_{layer_id}.dat"
        if os.path.exists(self.cache_k_path):
            os.remove(self.cache_k_path)
        
        if os.path.exists(self.cache_v_path):
            os.remove(self.cache_v_path)

        if os.path.exists(self.mask_bs_path):
            os.remove(self.mask_bs_path)

        if os.path.exists(self.score_bs_path):
            os.remove(self.score_bs_path)

        if os.path.exists(self.mask_est_path):
            os.remove(self.mask_est_path)
        
        self.max_batch_size = 1

        self.cache_q = np.memmap(
            self.cache_q_path,
            dtype="float32",
            mode="w+",
            shape=(
                self.max_batch_size,
                self.max_seq_len,
                self.num_heads,
                self.head_dim,
            ),
        )

        self.cache_k = np.memmap(
            self.cache_k_path,
            dtype="float32",
            mode="w+",
            shape=(
                self.max_batch_size,
                self.max_seq_len,
                self.num_kv_heads,
                self.head_dim,
            ),
        )
        self.cache_v = np.memmap(
            self.cache_v_path,
            dtype="float32",
            mode="w+",
            shape=(
                self.max_batch_size,
                self.max_seq_len,
                self.num_kv_heads,
                self.head_dim,
            ),
        )

        self.cache_q[:] = 0.0
        self.cache_k[:] = 0.0
        self.cache_v[:] = 0.0

        self.vertical_topk = torch.zeros((self.max_batch_size, self.num_heads, 1000), dtype=torch.int32) #1000
        self.slash = torch.zeros((self.max_batch_size, self.num_heads, 6096), dtype=torch.int32) #6096
        self.sparse_mask = None
        self.sparse_keys = None

        self.bs_idx = {i: [] for i in range(self.num_heads)}

        with open("distribution.json", "r") as f:
            self.main_dict = json.load(f)
        
        self.distribution_dict = self.main_dict[str(layer_id)]

    def qkv_cache_update(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        x_shape = x.shape
        assert len(x_shape) == 3

        batch_size, input_len, _ = x_shape
        x = torch.tensor(x)
        bsz, seqlen, _ = x.shape

        # Split data for CPU and XPU computation
        split_dim = x.shape[1] // 3  # Split 1/3rd on CPU, 2/3rd on XPU
        x_cpu, x_xpu = x[:, :split_dim, :], x[:, split_dim:, :]

        # Transfer data to respective devices
        x_cpu = x_cpu.to("cpu").contiguous()
        x_xpu = x_xpu.to("xpu").contiguous()
        freqs_cis_xpu = freqs_cis.to("xpu")

        # Efficiently duplicate self.wk and self.wv for CPU computation
        # wk_cpu = copy.deepcopy(self.wk).to("cpu")
        # wv_cpu = copy.deepcopy(self.wv).to("cpu")
        w_qkv_cpu = self.qkv_proj
        w_qkv_xpu = copy.deepcopy(self.qkv_proj).to("xpu")

        # Define async tasks for CPU and XPU computation
        def compute_on_cpu():
            nonlocal qkv_cpu
            qkv_cpu = self.qkv_proj(x_cpu)

        def compute_on_xpu():
            nonlocal qkv_xpu
            qkv_xpu = w_qkv_xpu(x_xpu)

        # Thread creation
        qkv_cpu, qkv_xpu = None, None
        thread_cpu = threading.Thread(target=compute_on_cpu)
        thread_xpu = threading.Thread(target=compute_on_xpu)

        # Start and join threads
        thread_cpu.start()
        thread_xpu.start()
        thread_cpu.join()
        thread_xpu.join()

        xq_cpu, xk_cpu, xv_cpu = qkv_cpu.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        xq_xpu, xk_xpu, xv_xpu = qkv_xpu.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Combine results before reshaping
        xq_combined = torch.cat([xq_cpu.to("cpu"), xq_xpu.to("cpu")], dim=1)
        xk_combined = torch.cat([xk_cpu.to("cpu"), xk_xpu.to("cpu")], dim=1)
        xv_combined = torch.cat([xv_cpu.to("cpu"), xv_xpu.to("cpu")], dim=1)

        # Reshape after concatenation
        xq_combined = xq_combined.view(batch_size, -1, self.num_heads, self.head_dim)
        xk_combined = xk_combined.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv_combined = xv_combined.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq_combined = self.query_norm(xq_combined)
            xk_combined = self.key_norm(xk_combined)

        # Apply rotary embeddings
        xq_combined = apply_rotary_emb(xq_combined, freqs_cis=freqs_cis.to("cpu"))
        xk_combined = apply_rotary_emb(xk_combined, freqs_cis=freqs_cis.to("cpu"))

        # Cache update
        self.cache_q[:batch_size, start_pos : start_pos + seqlen] = xq_combined.detach().cpu().numpy()
        self.cache_k[:batch_size, start_pos : start_pos + seqlen] = xk_combined.detach().cpu().numpy()
        self.cache_v[:batch_size, start_pos : start_pos + seqlen] = xv_combined.detach().cpu().numpy()

        # Clean up
        del x, qkv_cpu, qkv_xpu, xq_cpu, xq_xpu, xk_cpu, xv_cpu, xk_xpu, xv_xpu, xq_combined, xk_combined, xv_combined, w_qkv_xpu
        torch.xpu.empty_cache()

    def mask_generation(
        self,
        x: torch.Tensor,
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
        del x
        torch.xpu.empty_cache()
        xq = self.cache_q[:bsz,:seqlen]
        xq = torch.tensor(xq, device="xpu")
        if xq.shape[0] != 1:
            xq = xq.unsqueeze(0)
        xq = xq.transpose(1, 2)
        xq.mul_(self.scaling)
        torch.xpu.empty_cache()
        head_type = "BS"
        
        def vs_as_par(xq, bsz=bsz, seqlen=seqlen):
            
            #bsz, seqlen, _ = x.shape
            idx = min(64, seqlen)
            q = xq[:,:,:idx, :].to("xpu")
            
            chunk_size = 8000
            scores = None
            for i in range(0, prompt_len, chunk_size):
                start = i
                end = min(i + chunk_size, prompt_len)
                keys = self.cache_k[:bsz, start : end]
                keys = torch.tensor(keys).to("xpu")
                if keys.shape[0] != 1:
                    keys = keys.unsqueeze(0)
                if self.num_kv_heads != self.num_heads:
                    # [batch_size, max_seq_len, n_local_heads, head_dim]
                    keys = torch.repeat_interleave(keys, self.num_queries_per_kv, dim=2)
                
                keys = keys.transpose(1, 2)
                if scores is None:
                    scores = (torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)).to("xpu")
                else:
                    new_scores = (torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)).to("xpu")
                    scores = torch.cat([scores, new_scores], dim=-1)
                    del new_scores
                del keys            
                torch.xpu.empty_cache()

            arange = torch.arange(idx, device="xpu")
            idx_MASK = arange[None, None, :, None] >= arange[None, None, None, :]
            scores[:, :, :, -idx:] = torch.where(idx_MASK[...,-idx:,-idx:].to("xpu"), scores[:, :, :, -idx:], -torch.inf)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            
            
            torch.xpu.empty_cache()
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
            torch.xpu.empty_cache()
            self.slash = (prompt_len - 1) - torch.topk(self.slash, slash_size, -1).indices
            self.slash = self.slash.to("cpu")
            print("slash and vertical indices ready")
            self.sparse_mask, self.sparse_keys = create_sparse_causal_mask_all_heads(prompt_len, self.vertical_topk, self.slash, "cpu")
        
        def bs_par(xq, head_idx, mask_bs, score_bs):
            
            print("Hi Enter BS")
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
            
            print("Mask size: ", mask_est.shape)
            mask_est = mask_est.view(self.max_seq_len, self.max_seq_len)
            #mask_est.copy_(torch.tril(mask_est))
            idx = torch.arange(self.max_seq_len)
            mask =  idx[:, None] >= idx[None, :]
            print("Mask.shape: ", mask.shape)
            
            mask_est &= mask
            del mask
            print("Tril done")
            #mask_est.copy_(mask_est == 0)

            
            mask_est = mask_est[:q_len, :q_len]
            print("mask shape: ", mask_est.shape)

            row_mask = mask_est.all(dim=1)
            unique_rows = torch.nonzero(row_mask).squeeze(1)

            print("Row done")

            col_mask = mask_est.all(dim=0)
            unique_cols = torch.nonzero(col_mask).squeeze(1)

            print("col done")

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

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == 'xpu':
                    print(f"Unfreed XPU tensor: {obj.size()}")
            except Exception:
                pass

        print("mask creation complete")
        print("Mask Time = ", end - start)
        
    # def mask_generation(
    #     self,
    #     x: torch.Tensor,
    #     chunk_size: int,
    #     prompt_len: int,
    # ):
        
    #     print("enter mask creation")
    #     start = time.time()
    #     x = torch.tensor(x)
    #     bsz, seqlen, _ = x.shape
    #     # x = x.to("xpu")
        
    #     # xq = self.wq(x)
    #     # xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    #     # xq = apply_rotary_emb_q(xq, freqs_cis=freqs_cis.to("xpu"))
    #     # xq = xq.transpose(1, 2)

    #     xq = self.cache_q[:bsz,:seqlen]
    #     xq = torch.tensor(xq)
    #     if xq.shape[0] != 1:
    #         xq = xq.unsqueeze(0)
    #     xq = xq.transpose(1, 2)
    #     xq.mul_(self.scaling)
    #     head_type = "BS"
    #     def vs_as_par(xq):
            
    #         bsz, seqlen, _ = x.shape
    #         idx = min(64, seqlen)
    #         q = xq[:,:,:idx, :].to("xpu")
            
    #         chunk_size = 8000
    #         scores = None
    #         for i in range(0, prompt_len, chunk_size):
    #             start = i
    #             end = min(i + chunk_size, prompt_len)
    #             keys = self.cache_k[:bsz, start : end]
    #             keys = torch.tensor(keys).to("xpu")
    #             if keys.shape[0] != 1:
    #                 keys = keys.unsqueeze(0)
    #             if self.num_kv_heads != self.num_heads:
    #                 # [batch_size, max_seq_len, n_local_heads, head_dim]
    #                 keys = torch.repeat_interleave(keys, self.num_queries_per_kv, dim=2)
                
    #             keys = keys.transpose(1, 2)
    #             if scores is None:
    #                 scores = (torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)).to("xpu")
    #             else:
    #                 new_scores = (torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)).to("xpu")
    #                 scores = torch.cat([scores, new_scores], dim=-1)
    #                 del new_scores
    #             del keys            
    #             torch.xpu.empty_cache()

    #         arange = torch.arange(idx, device="xpu")
    #         idx_MASK = arange[None, None, :, None] >= arange[None, None, None, :]
    #         scores[:, :, :, -idx:] = torch.where(idx_MASK[...,-idx:,-idx:].to("xpu"), scores[:, :, :, -idx:], -torch.inf)
    #         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            
            
    #         torch.xpu.empty_cache()
    #         vertical = scores.sum(-2, keepdim=True)
    #         vertical_size, slash_size  = min(prompt_len, max(300, 30)), min(prompt_len, max(800, 50)) #  CHANGE HERE AS WELL
    #         self.vertical_topk = self.vertical_topk[:, :, :vertical_size]
    #         self.vertical_topk = (torch.topk(vertical, vertical_size, -1).indices)
    #         self.vertical_topk = self.vertical_topk.to("cpu")

    #         self.slash = self.slash[:, :, :slash_size]
    #         self.slash = sum_all_diagonal_matrix(scores)[...,:-idx + 1]
    #         del scores
    #         del idx_MASK
    #         del arange
    #         del q
    #         torch.xpu.empty_cache()
    #         self.slash = (prompt_len - 1) - torch.topk(self.slash, slash_size, -1).indices
    #         self.slash = self.slash.to("cpu")
    #         print("slash and vertical indices ready")
    #         self.sparse_mask, self.sparse_keys = create_sparse_causal_mask_all_heads(prompt_len, self.vertical_topk, self.slash, "cpu")
        
    #     def bs_par(xq, head_idx, mask_bs, score_bs):
            
    #         heads_per_kv = self.num_heads //self.num_kv_heads
            
    #         q = xq[:,head_idx:head_idx+1,:,:]
            
    #         block_size = 64
    #         q_len = seqlen
    #         block_num = (q_len - 1) // block_size + 1
    #         block_q = torch.zeros(1,1,block_num * block_size, self.head_dim).to(xq)
            
    #         block_q[:,:,:q_len] = q
            
    #         block_q = block_q.reshape(1,1,block_num,block_size,-1).mean(-2)
            
    #         batch_indices = np.arange(bsz)[:, None, None] 
    #         scores = None
    #         chunk_size = 2000
            
    #         for i in range(0, prompt_len, chunk_size):
                
    #             start = i
    #             end = min(i + chunk_size, prompt_len)
    #             keys = self.cache_k[:bsz, np.arange(start, end)[None, :, None], np.arange(head_idx // heads_per_kv, head_idx // heads_per_kv + 1)[None, :, None],:]
    #             keys = torch.tensor(keys) #.to("xpu")
    #             keys = keys.squeeze(-2)
                
    #             if keys.shape[0] != 1:
    #                 keys = keys.unsqueeze(0)
    #             block_num_k = ((end-start) - 1) // block_size + 1
                
    #             block_k= torch.zeros(1,1,block_num_k * block_size, self.head_dim).to(keys)
                                
    #             block_k[:,:,:(end-start)] = keys
    #             block_k = block_k.reshape(1,1,block_num_k,block_size,-1).mean(-2)
                
    #             scores = (torch.matmul(block_q, block_k.transpose(2, 3)) / math.sqrt(self.head_dim)) #.to("xpu")
                
    #             prev_size = (i - 1)// block_size + 1                
                
    #             score_bs[0:block_q.shape[-2], prev_size:prev_size + block_k.shape[-2]] = scores.detach().cpu().numpy() + mask_bs[0:block_q.shape[-2], prev_size:prev_size+block_k.shape[-2]]  
                
    #             del keys, block_k, scores            
    #             torch.xpu.empty_cache()
                
            
    #         topk_values, topk_indices = torch.topk(-torch.tensor(score_bs[0:block_num, 0:block_num]), 100, dim=-1)
            
    #         block_rows = torch.arange(block_num, device="cpu").view(-1, 1).repeat(1, block_num)
    #         block_cols = torch.arange(block_num, device="cpu").view(1, -1).repeat(block_num, 1)
            
    #         block_mask = torch.ones((block_num, block_num), dtype=torch.int32, device="cpu")
    #         block_mask[torch.arange(block_num, device="cpu").view(-1, 1), topk_indices] = 0
            
    #         mask_est = block_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, block_size, block_size)
         
    #         mask_est.copy_(mask_est[:q_len, :q_len])
            
    #         del block_mask, block_rows, block_cols, block_q,topk_indices, topk_values, q
    #         torch.xpu.empty_cache()
            
            
    #         mask_est.copy_(torch.tril(mask_est))
    #         mask_est.copy_(mask_est == 0)
            
    #         mask_flat = mask_est.flatten()
    #         mask_est2_size = mask_est.size(1)
    #         del mask_est
    #         torch.xpu.empty_cache()

    #         nonzero_indices = (torch.where(mask_flat == False)[0])
    #         nonzero_indices = nonzero_indices1
    #         del mask_flat
    #         torch.xpu.empty_cache()
            
    #         rows = (nonzero_indices // mask_est2_size)
            
    #         row_counts = torch.bincount(rows)
    #         unique_rows = torch.nonzero(row_counts).squeeze(1)

    #         del rows, row_counts
    #         torch.xpu.empty_cache()

    #         cols = nonzero_indices % mask_est2_size
            
    #         col_counts = torch.bincount(cols)
    #         unique_cols = torch.nonzero(col_counts).squeeze(1)
            
    #         del cols, col_counts
    #         torch.xpu.empty_cache()
            
    #         del nonzero_indices
    #         torch.xpu.empty_cache()
    #         self.bs_idx[head_idx].append((unique_rows))
            
    #         self.bs_idx[head_idx].append((unique_cols))
            
    #         del unique_cols, unique_rows
    #         gc.collect()
    #         torch.xpu.empty_cache()
            
         
    #     count_bs = 0
    #     idx_list = []
    #     for sub_key, value in self.distribution_dict.items():
    #         if value[0] == "BS":
    #             count_bs += 1
    #             idx_list.append(int(sub_key))
    #     streams = [torch.xpu.Stream() for _ in range(count_bs % 2 + 1)]
        
    #     with torch.xpu.stream(streams[count_bs % 2 ]):
    #         vs_as_par(xq)

    #     mask_bs = np.memmap(
    #         self.mask_bs_path,
    #         dtype="int32",
    #         mode="w+",
    #         shape=(
    #             self.max_seq_len,
    #             self.max_seq_len,
    #         ),
    #     )

    #     score_bs = np.memmap(
    #         self.score_bs_path,
    #         dtype="float32",
    #         mode="w+",
    #         shape=(
    #             self.max_seq_len,
    #             self.max_seq_len,
    #         ),
    #     )
    #     for i in range(self.max_seq_len):
    #         mask_bs[i, :i + 1] = 1

    #     # Flush data to ensure it's written to disk
    #     mask_bs.flush()

    #     for idx in idx_list:
    #         with torch.xpu.stream(streams[idx % 1]):
    #             bs_par(xq, idx, mask_bs, score_bs)
        
    #     torch.xpu.synchronize()


    #     del xq , mask_bs, score_bs

    #     end = time.time()
    #     torch.xpu.empty_cache()
    #     gc.collect()

    #     if os.path.exists(self.mask_bs_path):
    #         os.remove(self.mask_bs_path)

    #     if os.path.exists(self.score_bs_path):
    #         os.remove(self.score_bs_path)

    #     print("mask creation complete")
    #     print("Mask Time = ", end - start)
    
    def prefill_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        chunk_size: int,
        prompt_len: int,
    ):
        
        bsz, seqlen, _ = x.shape
        
        if torch.isnan(x).any():
            print("NaN detected in x!")
        
        xq = self.cache_q[:bsz,:seqlen]
        xq = torch.tensor(xq, device = "xpu")
        if xq.shape[0] != 1:
            xq = xq.unsqueeze(0)
        xq = xq.transpose(1, 2)
        xq.mul_(self.scaling)

        del x
        torch.xpu.empty_cache()
        if torch.isnan(xq).any():
            print("NaN detected in xq!")

        # xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # xq = apply_rotary_emb_q(xq, freqs_cis=freqs_cis.to("xpu"))

        #xq = xq.transpose(1, 2) #.to("xpu")
        
        #xq.mul_(self.scaling)
        chunked_sparse_keys = self.sparse_keys.to("cpu").split(chunk_size)

        selective_indices = self.sparse_keys
        
        if torch.isnan(selective_indices).any():
            print("NaN detected in selective_indices!")

        bs = bsz

        heads_per_kv = self.num_heads //self.num_kv_heads

        scores = {i: None for i in range(self.num_heads)}
        
        num_sparse_keys = self.sparse_keys.size(-1)
        mask = None

        mask = torch.zeros((bsz, self.num_heads, seqlen, num_sparse_keys))
        
        seq_range = torch.arange(start_pos, start_pos + seqlen, device="xpu").view(1, 1, -1, 1)
        sparse_keys_expanded = self.sparse_keys.unsqueeze(2).expand(bsz, self.num_heads, seqlen, num_sparse_keys).to("xpu")
        seq_range_expanded = seq_range.expand(bsz, self.num_heads, seqlen, num_sparse_keys).to("xpu")

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
        
        

        output = {i: None for i in range(self.num_heads)}

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
                
                keys = torch.tensor(gathered_keys, device = device)
                
                keys = keys.transpose(1, 2)

                if torch.isnan(keys).any():
                    print("NaN detected in keys!", chunk)

                if scores[head_idx] == None:
                    scores[head_idx] = (torch.matmul(xq[:,head_idx:head_idx+1,:,:], keys.transpose(2, 3))).to(device)
                    
                else:
                    new_scores = torch.matmul(xq[:,head_idx:head_idx+1,:,:].to(device), keys.transpose(2, 3))                    
                    scores[head_idx] = torch.cat([scores[head_idx], new_scores], dim=-1)
                    
                    del new_scores
                del keys
                torch.xpu.empty_cache()

                if torch.isnan(scores[head_idx]).any():
                    print("NaN detected in scores!")
                    
            if self.attn_logit_softcapping is not None:
                scores[head_idx] = scores[head_idx] / self.attn_logit_softcapping
                scores[head_idx] = torch.tanh(scores)
                scores[head_idx] = scores[head_idx] * self.attn_logit_softcapping

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
                kv_head_indices = np.arange(self.num_kv_heads)[None, :, None]  # Shape: [1, n_kv_heads, 1]

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
        

        def par_heads_bs(head_idx, mask, device):
                        
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
            
            keys = torch.tensor(gathered_keys, device = device)  # Convert to PyTorch tensor
            
            keys = keys.transpose(1, 2)
            keys = keys.transpose(2, 3)
            keys = keys.squeeze(1)
            # print("keys shape ", keys.shape)
            if torch.isnan(keys).any():
                print("NaN detected in keys!", chunk)
            
            xq_sliced = (xq[:,head_idx,:,:])
            scores[head_idx] = (torch.bmm(xq_sliced, keys)).to(device)
            
            del keys, xq_sliced, mask_rows, mask_cols
            torch.xpu.empty_cache()
            scores[head_idx] = torch.tensor(scores[head_idx])
            if torch.isnan(scores[head_idx]).any():
                print("NaN detected in scores!")
            
            if self.attn_logit_softcapping is not None:
                scores[head_idx] = scores[head_idx] / self.attn_logit_softcapping
                scores[head_idx] = torch.tanh(scores)
                scores[head_idx] = scores[head_idx] * self.attn_logit_softcapping
            
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
            
            del mask
            del final_result
            del values
            del row_indices, valid_mask, filtered_rows, filtered_cols, scores[head_idx]
            torch.xpu.empty_cache()
            gc.collect()

        a = time.time()
                
        streams = [torch.xpu.Stream() for _ in range((32 // 12) + 1)]

        for sub_key, value in self.distribution_dict.items():
            if(value[1] == "xpu"):
                with torch.xpu.stream(streams[-1]):
                    if(value[0] == "VS" or value[0] == "A"):
                        par_heads_gpu(int(sub_key), mask[:, int(sub_key):int(sub_key)+1, :, :], value[1]) # THIS IS FOR VS
                    else:
                        par_heads_bs(int(sub_key), None, value[1])
            else:
                with torch.xpu.stream(streams[int(sub_key) // 12]):
                    if(value[0] == "VS" or value[0] == "A"):
                        par_heads_gpu(int(sub_key), mask[:, int(sub_key):int(sub_key)+1, :, :], value[1]) # THIS IS FOR VS
                    else:
                        par_heads_bs(int(sub_key), None, value[1])

        torch.xpu.synchronize()
        b = time.time()
        print("func time = ", b - a)
        output1 = torch.cat(list(output.values()), dim=1)
        output2 = output1.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = output2.to("cpu")
        print("Output.shape: ", output.shape)
        print("Out proj weight.shape", self.o_proj.weight.shape) 
        out = self.o_proj(output)
        del output
        del output1
        del output2
        del xq
        #del scores
        del streams
        
        torch.xpu.empty_cache()
        gc.collect()
        return out.to("cpu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,        
        chunk_size,
        start_pos,
        is_prefill,
        prompt_len,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        if is_prefill:
            return self.prefill_forward(hidden_states, start_pos, chunk_size, prompt_len)
        
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3
        print("Hidden shape in decode: ", hidden_states_shape)

        batch_size, input_len, _ = hidden_states_shape
        seqlen = input_len
        print("Seqlen in decode: ", seqlen)
        qkv = self.qkv_proj(hidden_states)
        print("QKV shape: ", qkv.shape)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)
        print("Xq.shape, xk.shape and xv.shape", xq.shape, xk.shape, xv.shape)
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        print("Xq.shape, xk.shape and xv.shape", xq.shape, xk.shape, xv.shape)
        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)
        print("Xq.shape, xk.shape and xv.shape", xq.shape, xk.shape, xv.shape)
        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        self.cache_k[:batch_size, start_pos : start_pos + seqlen] = xk.cpu().numpy()
        self.cache_v[:batch_size, start_pos : start_pos + seqlen] = xv.cpu().numpy()
        xq = xq.transpose(1, 2)
        xq.mul_(self.scaling)
        scores = None
        for i in range(0, start_pos + seqlen, chunk_size):
            start = i
            end = min(i + chunk_size, start_pos + seqlen)
            print("Start, end: ", start, end)
            key = self.cache_k[:batch_size, start : end]
            key = torch.tensor(key, device=xq.device)
            print("Keys.shape: ", key.shape)
            
            if key.shape[0] != 1:
                key = keys.unsqueeze(0)

            if self.num_kv_heads != self.num_heads:
                # [batch_size, max_seq_len, n_local_heads, head_dim]
                key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)

            key = key.transpose(1, 2)
            print("Keys.shape: ", key.shape)
            
            if scores is None:
                scores = torch.matmul(xq, key.transpose(2, 3))
            else:
                new_scores = torch.matmul(xq, key.transpose(2, 3))
                scores = torch.cat([scores, new_scores], dim=-1)
        
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        print("Score shape: ", scores.shape)  
        if mask is not None:
            print("mask\n", mask)
            print(mask.shape)
            print("Score shape: ", scores.shape)
            scores = scores + mask
        
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        

        output = None
        for i in range(0, start_pos + seqlen, chunk_size):
            start = i
            end = min(i + chunk_size, start_pos + seqlen)
            value = self.cache_v[:batch_size, start : end]
            value = torch.tensor(value, device=xq.device)
            if value.shape[0] != 1:
                value = value.unsqueeze(0)
            if self.num_kv_heads != self.num_heads:
                # [batch_size, max_seq_len, n_local_heads, head_dim]
                value = torch.repeat_interleave(value,
                                                self.num_queries_per_kv,
                                                dim=2)
            value = value.transpose(
                1, 2
            )
            
            if output is None:
                output = torch.matmul(scores[:, :, :, start:end], value)
                print("Output first.shape, ", output.shape)
            else:
                output += torch.matmul(scores[:, :, :, start:end], value)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        # k_cache, v_cache = kv_cache
        # k_cache.index_copy_(1, kv_write_indices, xk)
        # v_cache.index_copy_(1, kv_write_indices, xv)

        # key = k_cache
        # value = v_cache
        # if self.num_kv_heads != self.num_heads:
        #     # [batch_size, max_seq_len, n_local_heads, head_dim]
        #     key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
        #     value = torch.repeat_interleave(value,
        #                                     self.num_queries_per_kv,
        #                                     dim=2)

        # # [batch_size, n_local_heads, input_len, head_dim]
        # q = xq.transpose(1, 2)
        # # [batch_size, n_local_heads, max_seq_len, head_dim]
        # k = key.transpose(1, 2)
        # v = value.transpose(1, 2)

        # # [batch_size, n_local_heads, input_len, max_seq_len]
        # q.mul_(self.scaling)
        # scores = torch.matmul(q, k.transpose(2, 3))
        # if (
        #     self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
        #     and self.sliding_window_size is not None
        #     and local_mask is not None
        # ):
        #     mask = local_mask

        # if self.attn_logit_softcapping is not None:
        #     scores = scores / self.attn_logit_softcapping
        #     scores = torch.tanh(scores)
        #     scores = scores * self.attn_logit_softcapping

        # scores = scores + mask
        # scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # # [batch_size, n_local_heads, input_len, head_dim]
        # output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output1 = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        print(output1.shape)
        del output
        output = self.o_proj(output1)

        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        layer_id: int,
    ):
        super().__init__()
        self.attn_type = gemma_config.AttentionType.GLOBAL
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=self.attn_type, layer_id=layer_id)
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    # TODO(imayank): Decouple Gemma versions into separate files.
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
        chunk_size: int,
        start_pos,
        is_prefill,
        prompt_len,
    ) -> torch.Tensor:
        #layer(h, freqs_cis1, None, None, mask, local_mask, chunk_size, chunk, is_prefill, seqlen)
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            chunk_size = chunk_size,
            start_pos=start_pos,
            is_prefill=is_prefill,
            prompt_len=prompt_len,
        )
        print("Hi Decoder1, ", self.attn_type)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# class Gemma2DecoderLayer(nn.Module):
#     def __init__(
#         self,
#         config: gemma_config.GemmaConfig,
#         attn_type: gemma_config.AttentionType,
#     ):
#         super().__init__()
#         self.attn_type = attn_type
#         self.self_attn = GemmaAttention(
#             config=config,
#             attn_type=self.attn_type,
#         )
#         self.mlp = GemmaMLP(
#             hidden_size=config.hidden_size,
#             intermediate_size=config.intermediate_size,
#             quant=config.quant,
#         )
#         self.input_layernorm = RMSNorm(config.hidden_size,
#                                        eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(config.hidden_size,
#                                                 eps=config.rms_norm_eps)
#         self.pre_feedforward_layernorm = (
#             RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#             if config.use_pre_ffw_norm
#             else None
#         )
#         self.post_feedforward_layernorm = (
#             RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#             if config.use_post_ffw_norm
#             else None
#         )

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         freqs_cis: torch.Tensor,
#         kv_write_indices: torch.Tensor,
#         kv_cache: Tuple[torch.Tensor, torch.Tensor],
#         mask: torch.Tensor,
#         local_mask: torch.Tensor,
#     ) -> torch.Tensor:
#         # Self Attention
#         print("Hi Decoder2, ", self.attn_type)
#         residual = hidden_states
#         hidden_states = self.input_layernorm(hidden_states)
#         hidden_states = self.self_attn(
#             hidden_states=hidden_states,
#             freqs_cis=freqs_cis,
#             kv_write_indices=kv_write_indices,
#             kv_cache=kv_cache,
#             mask=mask,
#             local_mask=local_mask,
#         )
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = residual + hidden_states

#         # MLP
#         residual = hidden_states
#         if self.pre_feedforward_layernorm is not None:
#             hidden_states = self.pre_feedforward_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         if self.post_feedforward_layernorm is not None:
#             hidden_states = self.post_feedforward_layernorm(hidden_states)
#         hidden_states = residual + hidden_states

#         return hidden_states


class GemmaModel(nn.Module):

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_position_embeddings

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaDecoderLayer(config, i))
            elif config.architecture in (
                gemma_config.Architecture.GEMMA_2,
                gemma_config.Architecture.GEMMA_3,
            ):
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                #print(attn_type)
                self.layers.append(Gemma2DecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unknown architecture: {config.architecture}')
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.start_pos = 0
        self.mask_bs_path = rf".\mask_bs.dat"
        self.score_bs_path = rf".\score_bs.dat"

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
        print("Mask init")
        for i in range(self.max_seq_len):
            self.mask_bs[i, :i + 1] = 1
        print("Mask complete")
        # Flush data to ensure it's written to disk
        self.mask_bs.flush()
        print("Mask Flush")

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
        hidden_embed_map,
        prompt_len,
    ) -> torch.Tensor:

        chunk_size = 2000
        orig_chunk = chunk_size
        seqlen = prompt_len
        if hidden_embed_map is not None:
            count = 0
            for layer in self.layers:
                def process_chunk1(chunk):
                    end = min(chunk + chunk_size, seqlen)
                    
                    freqs_cis1 = freqs_cis.get(layer.attn_type)[chunk:end]
                    
                    h = torch.tensor(hidden_embed_map[:, chunk:end, :])
                    
                    layer.self_attn.qkv_cache_update(layer.input_layernorm(torch.tensor(h)), chunk, freqs_cis1)
                    
                with ThreadPoolExecutor(max_workers=32) as executor:
                    futures = {executor.submit(process_chunk1, chunk): chunk for chunk in range(0, seqlen, chunk_size)}

                for future in concurrent.futures.as_completed(futures):
                    chunk = futures[future]
                    try:
                        future.result()
                        
                    except Exception as e:
                        print(f"Chunk {chunk} failed with error: {e}")
                
                idx = seqlen
                h = hidden_embed_map[:,seqlen-idx:seqlen,:]
                print("h , embed=", h.shape, hidden_embed_map.shape)

                layer.self_attn.mask_generation(layer.input_layernorm(torch.tensor(h)), chunk_size, seqlen, self.mask_bs, self.score_bs)

                def process_chunk(chunk):
                    chunk_end = min(chunk + chunk_size, seqlen)

                    #freqs_cis = self.freqs_cis[chunk:chunk_end]
                    h = hidden_embed_map[:, chunk:min(chunk + chunk_size, seqlen),:]
                    h = torch.tensor(h)

                    #h = layer(h, chunk, freqs_cis, mask, chunk_size, True, seqlen)
                    h = layer(h, None, None, None, mask, local_mask, chunk_size, chunk, True, seqlen)

                    hidden_embed_map[:, chunk:min(chunk + chunk_size, seqlen),:] = h.detach().cpu().numpy()
                    
                    del h
                    torch.xpu.empty_cache()
                    

                # Launch threads
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    executor.map(process_chunk, range(0, seqlen, chunk_size))

                # for chunk in range(0, seqlen,chunk_size):
                #     #freqs_cis1 = freqs_cis[chunk:min(chunk + chunk_size, seqlen)]
                #     h = hidden_embed_map[:, chunk:min(chunk + chunk_size, seqlen),:]
                #     h = torch.tensor(h)
                #     h = layer(h, None, None, None, mask, local_mask, chunk_size, chunk, True, seqlen)                    
                #     hidden_embed_map[:, chunk:min(chunk + chunk_size, seqlen),:] = h.detach().cpu().numpy()
                #     del h
                #     torch.xpu.empty_cache()
                
                print("Embed for ", count)
                count += 1
            
            hidden_states = torch.tensor(hidden_embed_map[:, :seqlen,:])
            self.start_pos = seqlen
            print("self.start_pos", self.start_pos)
        else:
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(
                    hidden_states=hidden_states,
                    freqs_cis=freqs_cis.get(layer.attn_type),
                    kv_write_indices=kv_write_indices,
                    kv_cache=None,
                    mask=None,
                    local_mask=local_mask,
                    chunk_size=chunk_size,
                    start_pos=self.start_pos,
                    is_prefill=False,
                    prompt_len=seqlen,
                )
            hidden_states = self.norm(hidden_states)
            self.start_pos += 1
        return hidden_states


class GemmaForCausalLM(nn.Module):

  def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
    super().__init__()
    self.config = config
    assert config.hidden_size % config.num_attention_heads == 0

    max_seq_len = config.max_position_embeddings
    head_dim = config.head_dim
    vocab_size = config.vocab_size

    self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
    self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
    self.model = GemmaModel(config)
    self.sampler = Sampler(vocab_size, config)

    # Pre-compute rotary embedding table.
    if config.architecture == gemma_config.Architecture.GEMMA_3:
      if config.rope_wave_length is None:
        raise ValueError('rope_wave_length must be provided for Gemma3.')

      rope_lengths = config.rope_wave_length
      defaults = {
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 10_000,
            }

      for attn_type, name in [
                (gemma_config.AttentionType.LOCAL_SLIDING, 'local_freqs_cis'),
                (gemma_config.AttentionType.GLOBAL, 'global_freqs_cis'),
            ]:
        theta = rope_lengths.get(
                    attn_type, defaults[attn_type]
                )
        self._register_freqs_cis(name, head_dim, max_seq_len, theta=theta)

    else:
        self._register_freqs_cis('freqs_cis', head_dim, max_seq_len)
    
    self.embed_path = rf".\input_embed.dat"
    self.dim = config.hidden_size
    self.is_prefill = True

  def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000
    ):
    self.register_buffer(
            name, precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta)
        )
    

  @torch.no_grad()
  def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    _bsz, seqlen = input_token_ids.shape

    hidden_states = None
    embed = None
    if self.is_prefill:
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

        print("Initial Embeddings")
        chunk_size = 512
        orig_chunk = chunk_size
        
        for chunk in range(0, seqlen,chunk_size):
            hidden_states = self.embedder(input_token_ids[:, chunk:min(chunk + chunk_size, seqlen)])
            print("Hidden state shape: ", hidden_states.shape)
            normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
            print("Normalizer shape, ", normalizer.shape)
            inter = hidden_states * normalizer
            print("Inter Shape, ", inter.shape)
            print("embed size, ", embed[:, chunk:min(chunk + chunk_size, seqlen),:].shape)
            embed[:, chunk:min(chunk + chunk_size, seqlen),:] = inter
        
        self.is_prefill = False
    
    else:
        hidden_states = self.embedder(input_token_ids)
        print("Input shape,  ", hidden_states.shape)
        print("Input type,  ", hidden_states.dtype)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        
        
    freqs_cis = {}
    
    if self.config.architecture == gemma_config.Architecture.GEMMA_3:
      freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
                self.local_freqs_cis.index_select(0, input_positions)
            )
      freqs_cis[gemma_config.AttentionType.GLOBAL] = (
                self.global_freqs_cis.index_select(0, input_positions)
            )
    else:
      freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
                self.freqs_cis.index_select(0, input_positions)
            )
      freqs_cis[gemma_config.AttentionType.GLOBAL] = (
                self.freqs_cis.index_select(0, input_positions)
            )
    
    hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices= None, #kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
            hidden_embed_map=embed,
            prompt_len=seqlen,
        )

    #kv_write_indices = input_positions

    # [batch_size, input_len, hidden_size]
    #hidden_states = self.embedder(input_token_ids)
    #print("Input shape,  ", hidden_states.shape)
    #print("Input type,  ", hidden_states.dtype)
    # Gemma normalizes the embedding by sqrt(hidden_size).
    # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    #normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
    #hidden_states = hidden_states * normalizer

    
    embedder_weight = self.embedder.weight
    if self.config.quant:
      embedder_weight = (
                embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
    next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
    return next_tokens, logits

  def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
    """Generates responses for given prompts using Gemma model."""
    # If a single prompt is provided, treat it as a batch of 1.
    is_str_prompt = isinstance(prompts, str)
    if is_str_prompt:
      prompts = [prompts]

    batch_size = len(prompts)
    prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
    prompt_tokens = [p[:min(len(p), self.config.max_position_embeddings - 10)] for p in prompt_tokens]
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len
    assert max_seq_len <= self.config.max_position_embeddings
    print("self.config.sliding_window_size: ", self.config.sliding_window_size)
    # build KV caches
    # kv_caches = []
    # for _ in range(self.config.num_hidden_layers):
    #   size = (batch_size, max_seq_len, self.config.num_key_value_heads,
    #                 self.config.head_dim)
    #   dtype = self.config.get_dtype()
    #   k_cache = torch.zeros(size=size, dtype=dtype, device=device)
    #   v_cache = torch.zeros(size=size, dtype=dtype, device=device)
    #   kv_caches.append((k_cache, v_cache))

    # prepare inputs
    token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      self.tokenizer.pad_id, dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64)
    for i, p in enumerate(prompt_tokens):
      token_ids_tensor[i, :len(p)] = torch.tensor(p)
      input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
    token_ids_tensor = token_ids_tensor.to(device)
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
    input_positions_tensor = torch.arange(0, min_prompt_len,
                                              dtype=torch.int64).to(device)
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)

    local_mask_tensor = mask_tensor + torch.tril(
            torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
            diagonal=-self.config.sliding_window_size,
        ) if self.config.sliding_window_size else None
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    curr_local_mask_tensor = local_mask_tensor.index_select(
          2, input_positions_tensor
      ) if local_mask_tensor is not None else None
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)
    
    # Prefill up to min_prompt_len tokens, then treat other prefill as
    # decode and ignore output.
    print("Min prompt len: ", min_prompt_len)
    for i in range(max_seq_len - min_prompt_len):
      print("curr_mask_tensor")
      print(curr_mask_tensor, curr_mask_tensor.shape)
      next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches= None, #kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                local_mask=curr_local_mask_tensor,
            )

      curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
      curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
      output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
      token_ids_tensor.index_copy_(1, output_index, output_token_ids)

      input_token_ids_tensor = output_token_ids
      input_positions_tensor = output_index.unsqueeze(dim=-1)
      curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
      curr_local_mask_tensor = local_mask_tensor.index_select(
                2, input_positions_tensor
            ) if local_mask_tensor is not None else None
      output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
      output_index = output_index + 1

    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
      trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
      if self.tokenizer.eos_id in trimmed_output:
        eos_index = trimmed_output.index(self.tokenizer.eos_id)
        trimmed_output = trimmed_output[:eos_index]
      results.append(self.tokenizer.decode(trimmed_output))

    # If a string was provided as input, return a string as output.
    return results[0] if is_str_prompt else results

  def load_weights(self, model_path: str):
        if os.path.isfile(model_path):
            self.load_state_dict(
                torch.load(
                    model_path, mmap=True, weights_only=True,
                )['model_state_dict'],
                strict=False,
            )
        else:
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()
