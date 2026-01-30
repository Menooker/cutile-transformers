# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# modified from https://github.com/NVIDIA/TileGym/blob/86e24c2c97956c0fee4435296c8f2bced1a78f14/src/tilegym/ops/cutile/attention.py

import math
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
import cuda.tile_experimental as ct_experimental


INV_LOG_2 = 1.0 / math.log(2)

# Define type aliases for Constant integers and booleans
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# --- FMHA Kernel Implementation ---
@ct.kernel(occupancy=2)
def fmha_kernel(
    Q2,
    K,
    V,
    Out,
    qk_scale: float,
    input_pos: int,
    TILE_D: ConstInt,  # TILE_D = hidden_size
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
    EVEN_K: ConstBool,
):
    """
    cuTile kernel for Fused Multi-Head Attention (FMHA).
    Computes attention output for a specific batch item and head, using tiling and online softmax.
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)  # [TILE_M]
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)  # [TILE_N]
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load query tile for this batch, head, and M-chunk
    q = ct.load(Q2, index=(batch_idx, bid_x, head_idx, 0), shape=(1, TILE_M, 1, TILE_D)).reshape(
        (TILE_M, TILE_D)
    )  # [TILE_M, TILE_D]

    # Loop over k, v and update accumulator
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        # When kv pos could exceed q pos
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        # When kv pos could exceed k_seqlen
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        # --- Compute QK product ---
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]

        # --- Apply Causal Masking ---
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            # Out of bound mask
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            # Causal mask
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)  # [TILE_M, TILE_N]
            mask = ct.where(mask, 0.0, -math.inf)  # [TILE_M, TILE_N]
            qk += mask

        # --- Online Softmax Update ---
        # Moving qk_scale multiplication after reduce_max is to improve performance.
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [TILE_M, TILE_N]

        # Attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]
        # Update m_i and l_i
        l_i = l_i * alpha + l_ij  # [TILE_M, 1]
        # Scale acc
        acc = acc * alpha  # [TILE_M, TILE_N]

        # --- Compute PV product ---
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))  # [TILE_N, TILE_D]
        p = p.astype(Q2.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, TILE_N]
        m_i = m_ij  # [TILE_M, 1]

    # --- Final Normalization and Store ---
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, TILE_M, 1, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, bid_x, head_idx, 0), tile=acc)


def _fmha_autotune_configs():
    """
    Iterator of autotune configurations for FMHA kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        all_configs = [SimpleNamespace(TILE_M=TM, TILE_N=TN, num_ctas=1, occupancy=occupancy)
                for TM in [32, 64] for TN in [32, 64, 128] for occupancy in [1, 2, 4, 8]]
        for config in all_configs:
            yield config
    else:
        # sm100 (Blackwell)
        yield SimpleNamespace(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2)


def cutile_autotune_fmha(
    stream,
    q2,
    k,
    v,
    o,
    sm_scale,
    input_pos,
    hidden_size,
    num_heads,
    query_group_size,
    is_causal,
    EVEN_K,
    TILE_M=64,
    TILE_N=64,
):
    batch_size, q_len, _, _ = q2.shape
    grid = (math.ceil(q_len / TILE_M), batch_size * num_heads, 1)
    ct.launch(stream,
            grid,  # 1D grid of processors
            fmha_kernel,
            (
                q2,
                k,
                v,
                o,
                sm_scale,
                input_pos,
                hidden_size,
                num_heads,
                TILE_M,
                TILE_N,
                query_group_size,
                is_causal,
                EVEN_K,
            ))
    # ct_experimental.autotune_launch(
    #     stream,
    #     grid_fn=lambda cfg: (
    #         math.ceil(q_len / cfg.TILE_M),
    #         batch_size * num_heads,
    #         1,
    #     ),
    #     kernel=fmha_kernel,
    #     args_fn=lambda cfg: (
    #         q,
    #         k,
    #         v,
    #         o,
    #         sm_scale,
    #         input_pos,
    #         hidden_size,
    #         num_heads,
    #         cfg.TILE_M,
    #         cfg.TILE_N,
    #         query_group_size,
    #         is_causal,
    #         EVEN_K,
    #     ),
    #     hints_fn=lambda cfg: {
    #         "num_ctas": cfg.num_ctas,
    #         "occupancy": cfg.occupancy,
    #     },
    #     search_space=_fmha_autotune_configs,
    # )
    return o

_max_tile_n = max(cfg.TILE_N for cfg in _fmha_autotune_configs())
def fmha_prefill(q: torch.Tensor, k, v, scaling, is_causal=True):
    batch_size, num_heads, q_len, hidden_size = q.shape
    _, num_head_kv, k_len, _ = k.shape

    assert num_heads % num_head_kv == 0
    query_group_size = num_heads // num_head_kv
    q2 = q.transpose(1, 2)
    assert q2.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    # q = q.contiguous() if not q.is_contiguous() else q
    # k = k.contiguous() if not k.is_contiguous() else k
    # v = v.contiguous() if not v.is_contiguous() else v
    o = torch.empty_like(q2)

    input_pos = 0  # prefill, causal

    EVEN_K = (k_len % _max_tile_n) == 0
    return cutile_autotune_fmha(
        torch.cuda.current_stream(),
        q2,
        k,
        v,
        o,
        scaling,
        input_pos,
        hidden_size,
        num_heads,
        query_group_size,
        is_causal,
        EVEN_K,
    )

# specialized fmha_prefill for fast inference of qwen2.5-1.5b
def fmha_prefill_fast(stream, q: torch.Tensor, k, v, scaling, is_causal=True, TILE_M=64, TILE_N=64):
    batch_size, num_heads, q_len, hidden_size = q.shape
    q2 = q.transpose(1, 2)

    # assert num_heads % num_head_kv == 0
    query_group_size = 6
    # assert q2.is_contiguous()
    # assert k.is_contiguous()
    # assert v.is_contiguous()
    # q = q.contiguous() if not q.is_contiguous() else q
    # k = k.contiguous() if not k.is_contiguous() else k
    # v = v.contiguous() if not v.is_contiguous() else v
    o = torch.empty_like(q2)

    input_pos = 0  # prefill, causal

    EVEN_K = True
    grid = (math.ceil(q_len / TILE_M), batch_size * num_heads, 1)
    ct.launch(stream,
            grid,  # 1D grid of processors
            fmha_kernel,
            (
                q2,
                k,
                v,
                o,
                scaling,
                input_pos,
                hidden_size,
                num_heads,
                TILE_M,
                TILE_N,
                query_group_size,
                is_causal,
                EVEN_K,
            ))




if __name__ == "__main__":
    # qwen2.5-1.5b
    # q2.shape: torch.Size([1, 128, 12, 128]) k.shape: torch.Size([1, 2, 128, 128]) v.shape: torch.Size([1, 2, 128, 128])
    # o.shape: torch.Size([1, 128, 12, 128])
    # sm_scale: 0.08838834764831845 input_pos: 0 hidden_size: 128
    # num_heads: 12 query_group_size: 6 is_causal: True EVEN_K: True
    import torch
    q2 = torch.randn((1, 128, 12, 128), dtype=torch.float16, device='cuda')
    q = q2.transpose(1, 2)
    k = torch.randn((1, 2, 128, 128), dtype=torch.float16, device='cuda')
    v = torch.randn((1, 2, 128, 128), dtype=torch.float16, device='cuda')
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import time
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    module =SimpleNamespace(num_key_value_groups=6)
    # enable cuda.tile_experimental._autotuner logging for debug
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # ct_experimental._autotuner.logger.setLevel(logging.DEBUG)

    stream = torch.cuda.current_stream()
    o = fmha_prefill_fast(stream, q, k, v, scaling=0.08838834764831845, is_causal=True)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        o = fmha_prefill_fast(stream, q, k, v, scaling=0.08838834764831845, is_causal=True)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Time taken for fmha_prefill: {(end_time - start_time) * 1000/10} ms")

    position_ids = torch.arange(128, device='cuda')
    o = sdpa(module, q, k, v, is_causal=True, attention_mask=None, position_ids=position_ids, sliding_window=None, use_cache = True)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        o = sdpa(module, q, k, v, is_causal=True, attention_mask=None, position_ids=position_ids, sliding_window=None, use_cache = True)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Time taken for sdpa: {(end_time - start_time) * 1000/10} ms")