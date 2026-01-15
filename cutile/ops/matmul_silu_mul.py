
import cuda.tile as ct
from math import ceil
import torch

def silu_and_mul(x, y, approx: bool):
    '''
    SiLU(x) * y
    SiLU(x) = x / (1 + exp(-x))
    approx: whether to use approximate exp
    '''
    denom = ct.add(1, ct.exp(-x), flush_to_zero=True)
    rounding_mode = ct.RoundingMode.APPROX if approx else None
    sigmoid_x = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=rounding_mode)
    silu = ct.mul(x, sigmoid_x, flush_to_zero=True)
    return ct.mul(silu, y, flush_to_zero=True)

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

    result = silu_and_mul(c1, c2, approx)
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

ConstInt = ct.Constant[int]
@ct.kernel(occupancy=ct.ByTarget(sm_120=8))
def gemv_silu_mul_split_k_kernel(A, B1, B2, C, f32acc, COUNTS,
                          tn: ConstInt, tk: ConstInt,
                          SPLIT_K: ConstInt, approx: ct.Constant[bool]):
    GROUP_SIZE_M = 1
    M = 1
    N = B1.shape[1]
    bidx, bidy = 0, ct.bid(0)
    bidz = ct.bid(1)
    # pad tile A to fake_tm rows, to enable tensorcore
    fake_tm = 16
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(1, tk))
    num_tiles_n = ct.num_tiles(B1, axis=0, shape=(tn, tk))
    sum1 = ct.full((fake_tm, tn), 0, dtype=ct.float32)
    sum2 = ct.full((fake_tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    split_size = ct.cdiv(num_tiles_k, SPLIT_K)
    for k in range(bidz * split_size, bidz * split_size + split_size, 1):
        # tile a has only one effective row, of shape tk. It is not efficient to use TMA to load it.
        a = ct.load(A, index=(bidx, k), shape=(fake_tm, tk),
                    padding_mode=zero_pad, allow_tma=False).astype(dtype)
        b1 = ct.load(B1, index=(bidy, k), shape=(tn, tk),
                    padding_mode=zero_pad).astype(dtype)
        b1 = ct.transpose(b1)
        sum1 = ct.mma(a, b1, sum1)
        b2 = ct.load(B2, index=(bidy, k), shape=(tn, tk),
                    padding_mode=zero_pad).astype(dtype)
        b2 = ct.transpose(b2)
        sum2 = ct.mma(a, b2, sum2)
    # only the first row of sum is needed
    sum1 = ct.extract(sum1, index=(0, 0), shape=(1, tn))
    sum1 = ct.reshape(sum1, (tn,))
    sum2 = ct.extract(sum2, index=(0, 0), shape=(1, tn))
    sum2 = ct.reshape(sum2, (tn,))

    count_offset = ct.bid(0)
    C_offset = ct.arange(tn, dtype=ct.int32) + bidy * tn
    ct.atomic_add(f32acc, (0, C_offset), sum1)
    ct.atomic_add(f32acc, (1, C_offset), sum2)
    new_count = ct.atomic_add(COUNTS, count_offset, 1)
    if (new_count + 1) % SPLIT_K == 0:
        result1 = ct.gather(f32acc, (0, C_offset))
        result2 = ct.gather(f32acc, (1, C_offset))
        result = silu_and_mul(result1, result2, approx=approx).astype(C.dtype)

        ct.scatter(C, (0, C_offset), result.astype(C.dtype))
        ct.scatter(f32acc, (0, C_offset), 0)
        ct.scatter(COUNTS, count_offset, 0)

f32acc = None
counts = None
def launch_gemv_silu_mul(a: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor, c: torch.Tensor, approx: bool = True):
    stream = torch.cuda.current_stream()
    split_k, tile_n, tile_k = 16, 64, 64
    M, N, K = a.shape[0], c.shape[1], a.shape[1]
    grid = (ceil(N/tile_n), split_k, 1)
    assert b1.shape == (N, K)
    assert b2.shape == (N, K)
    assert M == 1

    f32acc = None
    counts = None
    if f32acc is None or f32acc.shape[1] < N:
        f32acc = torch.zeros((2, N), device='cuda', dtype=torch.float32)
    if counts is None or counts.shape[0] < grid[0]:
        counts = torch.zeros((grid[0],), device='cuda', dtype=torch.int32)
    args = (a, b1, b2, c, f32acc, counts, tile_n, tile_k, split_k, False)
    ct.launch(stream,
            grid,
            gemv_silu_mul_split_k_kernel,
            args)


def test_matmul():
    import time
    import torch
    # Create input data
    tile_m, tile_n, tile_k = 128, 64, 128
    M, N, K = 4096, 4096, 4096
    grid = (ceil(M/tile_m) * ceil(N/tile_n), 1, 1)
    print("Grid size:", grid)

    a = torch.rand((M, K), device='cuda', dtype=torch.float16)
    b1 = torch.rand((K, N), device='cuda', dtype=torch.float16)
    b2 = torch.rand((K, N), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float16)

    # Launch kernel

    #benchmark this

    stream = torch.cuda.current_stream()
    args = (a, b1, b2, c, tile_m, tile_n, tile_k, False)
    ct.launch(stream,
            grid,  # 1D grid of processors
            matmul_silu_mul,
            args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        ct.launch(stream,
                grid,  # 1D grid of processors
                matmul_silu_mul,
                args)
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"matmul_silu_mul kernel time for 10 runs: {total*1000:.4f} ms")

    c1 = torch.matmul(a, b1.T)
    c2 = torch.matmul(a, b2.T)
    expected = c1 * torch.sigmoid(c1) * c2
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        c1 = torch.matmul(a, b1.T)
        c2 = torch.matmul(a, b2.T)
        expected = c1 * torch.sigmoid(c1) * c2
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"pytorch kernel time for 10 runs: {total*1000:.4f} ms")
    torch.testing.assert_close(c, expected)

    print("✓ matmul_silu_mul passed!")


#qwen2 
#launch_matmul_silu_mul torch.Size([1, 1536]) torch.Size([8960, 1536])
def test_gemv_split_k():
    import time
    import torch
    M, N, K = 1, 8960, 1536
    split_k = 16
    a = torch.rand((M, K), device='cuda', dtype=torch.float16)
    b1 = torch.rand((N, K), device='cuda', dtype=torch.float16)
    b2 = torch.rand((N, K), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float16)

    stream = torch.cuda.current_stream()
    tile_n, tile_k = 64, 64
    grid = (ceil(N/tile_n), 1, split_k)
    f32acc = torch.zeros((2, N), device='cuda', dtype=torch.float32)
    counts = torch.zeros((grid[0],), device='cuda', dtype=torch.int32)
    args = (a, b1, b2, c, f32acc, counts, tile_n, tile_k, split_k, False)
    ct.launch(stream,
            grid,
            gemv_silu_mul_split_k_kernel,
            args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        ct.launch(stream,
                grid,
                gemv_silu_mul_split_k_kernel,
                args)
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"gemv_silu_mul_split_k kernel time for 10 runs: {total*1000:.4f} ms")

    c1 = torch.matmul(a, b1.T)
    c2 = torch.matmul(a, b2.T)
    expected = c1 * torch.sigmoid(c1) * c2
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        c1 = torch.matmul(a, b1.T)
        c2 = torch.matmul(a, b2.T)
        expected = c1 * torch.sigmoid(c1) * c2
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"pytorch kernel time for 10 runs: {total*1000:.4f} ms")
    torch.testing.assert_close(c, expected)

    print("✓ gemv_silu_mul_split_k passed!")


if __name__ == "__main__":
    test_matmul()
    test_gemv_split_k()
