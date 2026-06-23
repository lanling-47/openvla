"""
PagedAttention kernel — Triton implementation for paged KV cache.

Supports:
  - Prefill: PyTorch SDPA with causal/prefix masking
  - Decode: Triton-based paged attention for single-token decode
  - KV Store: Triton kernel for writing KV into paged cache slots

Ported from nano-vllm-project/attention.py.
"""

import torch
from torch import nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============================================================
# Triton kernel: Store KV into paged cache
# ============================================================

if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr, key_stride, value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
        D: tl.constexpr,
    ):
        """Write new KV pairs into their assigned cache slots."""
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """Store key/value tensors into paged KV cache at specified slots."""
    if not HAS_TRITON:
        raise RuntimeError("Triton required for store_kvcache")
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )


# ============================================================
# Prefill Attention: PyTorch SDPA (supports GQA + prefix cache)
# ============================================================

def prefill_attention_sdpa(q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
    """
    Prefill attention using PyTorch native SDPA.
    Handles GQA (grouped query attention) and prefix cache scenarios.

    Args:
        q: [total_q_tokens, num_heads, head_dim]
        k: [total_kv_tokens, num_kv_heads, head_dim]
        v: [total_kv_tokens, num_kv_heads, head_dim]
        cu_seqlens_q: cumulative sequence lengths for queries
        cu_seqlens_k: cumulative sequence lengths for keys
        scale: attention scale factor (1/sqrt(head_dim))
    """
    cu_q = cu_seqlens_q.tolist() if hasattr(cu_seqlens_q, 'tolist') else cu_seqlens_q
    cu_k = cu_seqlens_k.tolist() if hasattr(cu_seqlens_k, 'tolist') else cu_seqlens_k
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_heads // num_kv_heads

    outputs = []
    for i in range(len(cu_q) - 1):
        q_start, q_end = cu_q[i], cu_q[i + 1]
        k_start, k_end = cu_k[i], cu_k[i + 1]
        seq_q = q[q_start:q_end]
        seq_k = k[k_start:k_end]
        seq_v = v[k_start:k_end]

        # GQA: repeat KV heads
        if group_size > 1:
            seq_k = seq_k.repeat_interleave(group_size, dim=1)
            seq_v = seq_v.repeat_interleave(group_size, dim=1)

        # Reshape for SDPA: [1, H, S, D]
        seq_q = seq_q.transpose(0, 1).unsqueeze(0)
        seq_k = seq_k.transpose(0, 1).unsqueeze(0)
        seq_v = seq_v.transpose(0, 1).unsqueeze(0)

        sq, sk = seq_q.shape[2], seq_k.shape[2]
        if sq < sk:
            # Prefix cache: first (sk-sq) keys visible, rest causal
            prefix_mask = torch.ones(sq, sk - sq, device=q.device, dtype=torch.bool)
            causal_part = torch.tril(torch.ones(sq, sq, device=q.device, dtype=torch.bool))
            attn_mask = torch.cat([prefix_mask, causal_part], dim=1)
        else:
            attn_mask = torch.tril(torch.ones(sq, sk, device=q.device, dtype=torch.bool))

        o = F.scaled_dot_product_attention(seq_q, seq_k, seq_v, attn_mask=attn_mask,
                                           scale=scale, is_causal=False)
        outputs.append(o.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)


# ============================================================
# Decode Attention: PyTorch reference (paged KV cache)
# ============================================================

def decode_attention_paged(q, k_cache, v_cache, block_tables, context_lens,
                           num_heads, num_kv_heads, kv_block_size, scale):
    """
    Decode attention with paged KV cache (PyTorch reference).
    For production, use the custom CUDA kernel (paged_attention_ext).

    Args:
        q: [batch_size, num_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: [batch_size, max_num_blocks]
        context_lens: [batch_size]
    """
    batch_size = q.shape[0]
    head_dim = q.shape[2]
    group_size = num_heads // num_kv_heads
    outputs = []

    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        bt = block_tables[b]
        num_blocks_needed = (ctx_len + kv_block_size - 1) // kv_block_size

        k_list, v_list = [], []
        for blk in range(num_blocks_needed):
            blk_id = bt[blk].item()
            tokens_in_blk = min(kv_block_size, ctx_len - blk * kv_block_size)
            k_list.append(k_cache[blk_id, :tokens_in_blk])
            v_list.append(v_cache[blk_id, :tokens_in_blk])

        k_seq = torch.cat(k_list, dim=0)  # [ctx_len, num_kv_heads, head_dim]
        v_seq = torch.cat(v_list, dim=0)

        q_vec = q[b]  # [num_heads, head_dim]
        if group_size > 1:
            k_seq = k_seq.repeat_interleave(group_size, dim=1)
            v_seq = v_seq.repeat_interleave(group_size, dim=1)

        scores = torch.einsum('hd,thd->ht', q_vec.float(), k_seq.float()) * scale
        probs = torch.softmax(scores, dim=1)
        o = torch.einsum('ht,thd->hd', probs, v_seq.float())
        outputs.append(o.to(q.dtype))

    return torch.stack(outputs, dim=0)
