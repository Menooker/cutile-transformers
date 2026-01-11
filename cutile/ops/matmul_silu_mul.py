
import cuda.tile as ct
from math import ceil
import torch

@ct.kernel(occupancy=ct.ByTarget(sm_120=8))
def matmul_silu_mul(a, b1, b2, c, TILE_M: ct.Constant[int], TILE_N: ct.Constant[int], TILE_K: ct.Constant[int], approx: ct.Constant[bool]):
    '''
    perform C = SiLU(A @ B1^T) * (A @ B2^T)
    '''
    pid = ct.bid(0)
    M,K = a.shape[0], a.shape[1]
    N = b1.shape[0]
    num_tiles_m = ct.cdiv(M, TILE_M)
    num_tiles_n = ct.cdiv(N, TILE_N)
    num_tiles_k = ct.cdiv(K, TILE_K)
    m_idx = pid // num_tiles_n
    n_idx = pid % num_tiles_n
    c1 = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    c2 = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    padding_mode = ct.PaddingMode.ZERO

    dtype = ct.bfloat16
    for k in range(num_tiles_k):
        a_tile = ct.load(a, index=(m_idx, k), shape=(TILE_M, TILE_K), padding_mode=padding_mode)
        b1_tile = ct.load(b1, index=(n_idx, k), shape=(TILE_N, TILE_K), padding_mode=padding_mode)
        b1_tile = ct.transpose(b1_tile)
        c1 = ct.mma(a_tile, b1_tile, c1)
        b2_tile = ct.load(b2, index=(n_idx, k), shape=(TILE_N, TILE_K), padding_mode=padding_mode)
        b2_tile = ct.transpose(b2_tile)
        c2 = ct.mma(a_tile, b2_tile, c2)


    denom = ct.add(1, ct.exp(-c1), flush_to_zero=True)
    rounding_mode = ct.RoundingMode.APPROX if approx else None
    sigmoid_c1 = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=rounding_mode)

    # Perform SiLU(c1) * c2
    silu_c1 = ct.mul(c1, sigmoid_c1, flush_to_zero=True)
    result = ct.mul(silu_c1, c2, flush_to_zero=True)

    result = result.astype(c.dtype)
    # Store result
    ct.store(c, index=(m_idx, n_idx), tile=result)

def launch_matmul_silu_mul(a: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor, c: torch.Tensor, approx: bool = True):
    stream = torch.cuda.current_stream()
    tile_m, tile_n, tile_k = 128, 64, 64
    M, N, K = a.shape[0], c.shape[1], a.shape[1]
    grid = (ceil(M/tile_m) * ceil(N/tile_n), 1, 1)
    assert b1.shape == (N, K)
    assert b2.shape == (N, K)
    ct.launch(stream,
            grid,  # 1D grid of processors
            matmul_silu_mul,
            (a, b1, b2, c, tile_m, tile_n, tile_k, approx))

#qwen2 
#launch_matmul_silu_mul torch.Size([1, 1536]) torch.Size([8960, 1536])

def test():
    import time
    import torch
    # Create input data
    tile_m, tile_n, tile_k = 128, 64, 64
    M, N, K = 4096, 4096, 4096
    grid = (ceil(M/tile_m) * ceil(N/tile_n), 1, 1)
    print("Grid size:", grid)

    a = torch.rand((M, K), device='cuda', dtype=torch.bfloat16)
    b1 = torch.rand((K, N), device='cuda', dtype=torch.bfloat16)
    b2 = torch.rand((K, N), device='cuda', dtype=torch.bfloat16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.bfloat16)

    # Launch kernel

    #benchmark this

    stream = torch.cuda.current_stream()
    args = (a, b1, b2, c, tile_m, tile_n, tile_k, False)
    ct.launch(stream,
            grid,  # 1D grid of processors
            matmul_silu_mul,
            args)
    start = time.time()
    for _ in range(10):
        ct.launch(stream,
                grid,  # 1D grid of processors
                matmul_silu_mul,
                args)
    torch.cuda.synchronize()
    end = time.time()
    print(f"matmul_silu_mul kernel time for 10 runs: {end - start:.4f} seconds")

    c1 = torch.matmul(a, b1.T)
    c2 = torch.matmul(a, b2.T)
    expected = c1 * torch.sigmoid(c1) * c2
    torch.testing.assert_close(c, expected)

    print("✓ matmul_silu_mul passed!")


if __name__ == "__main__":
    test()
