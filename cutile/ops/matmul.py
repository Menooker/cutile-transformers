
import cuda.tile as ct
from math import ceil
import torch

ACT_RELU = 1
ACT_SILU = 2
ACT_NONE = 0

ConstInt = ct.Constant[int]

@ct.kernel(occupancy=ct.ByTarget(sm_120=8))
def gemv_split_k_kernel(A, B, C, LOCKS, COUNTS,
                          tn: ConstInt, tk: ConstInt,
                          SPLIT_K: ConstInt):
    GROUP_SIZE_M = 1
    M = 1
    N = B.shape[1]
    bidx, bidy = 0, ct.bid(0)
    bidz = ct.bid(1)
    fake_tm = 16
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(1, tk))
    sum = ct.full((fake_tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    split_size = ct.cdiv(num_tiles_k, SPLIT_K)
    for k in range(bidz * split_size, bidz * split_size + split_size, 1):
        a = ct.load(A, index=(bidx, k), shape=(fake_tm, tk),
                    padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(bidy, k), shape=(tn, tk),
                    padding_mode=zero_pad).astype(dtype)
        b = ct.transpose(b)
        sum = ct.mma(a, b, sum)
    sum = ct.extract(sum, index=(0, 0), shape=(1, tn))
    sum = ct.reshape(sum, (tn,))
    sum = ct.astype(sum, C.dtype)
    lock_offset = ct.bid(0)
    count_offset = lock_offset
    C_offset = ct.arange(tn, dtype=ct.int32)
    C_offset = C_offset + bidy * tn
    C_offset_0 = ct.full((tn), 0, dtype=ct.int32)
    while ct.atomic_cas(LOCKS, lock_offset, 0, 1, memory_order=ct.MemoryOrder.ACQUIRE) == 1:
        pass
    count = ct.gather(COUNTS, count_offset)
    if count == 0:
        ct.scatter(C, (C_offset_0, C_offset), sum)
        #ct.store(C, index=(bidx, bidy), tile=sum)
    else:
        # curr = ct.load(C, index=(bidx, bidy), shape=(1, tn))
        # ct.store(C, index=(bidx, bidy), tile=(curr + sum))
        curr = ct.gather(C, (C_offset_0, C_offset))
        ct.scatter(C, (C_offset_0, C_offset), curr + sum)
    ct.scatter(COUNTS, count_offset, (count + 1) % SPLIT_K)
    ct.atomic_xchg(LOCKS, lock_offset, 0, memory_order=ct.MemoryOrder.RELEASE)


@ct.kernel(occupancy=ct.ByTarget(sm_120=8))
def matmul(a, b, c, TILE_M: ct.Constant[int], TILE_N: ct.Constant[int], TILE_K: ct.Constant[int], transb: ct.Constant[bool], act: ct.Constant[int]):
    # Get the 1D pid
    pid = ct.bid(0)
    M,K = a.shape[0], a.shape[1]
    N = b.shape[1] if not transb else b.shape[0]
    num_tiles_m = ct.cdiv(M, TILE_M)
    num_tiles_n = ct.cdiv(N, TILE_N)
    num_tiles_k = ct.cdiv(K, TILE_K)
    m_idx = pid // num_tiles_n
    n_idx = pid % num_tiles_n
    accumulator = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    padding_mode = ct.PaddingMode.ZERO

    dtype = ct.bfloat16
    for k in range(num_tiles_k):
        a_tile = ct.load(a, index=(m_idx, k), shape=(TILE_M, TILE_K), padding_mode=padding_mode)
        if transb:
            b_tile = ct.load(b, index=(n_idx, k), shape=(TILE_N, TILE_K), padding_mode=padding_mode)
            b_tile = ct.transpose(b_tile)
        else:
            b_tile = ct.load(b, index=(k, n_idx), shape=(TILE_K, TILE_N), padding_mode=padding_mode)
        accumulator = ct.mma(a_tile, b_tile, accumulator)
    if act == 1:
        accumulator = ct.maximum(accumulator, 0)
    elif act == 2:
        accumulator = accumulator * ct.sigmoid(accumulator)
    # Load input tiles
    accumulator = accumulator.astype(c.dtype)
    # Store result
    ct.store(c, index=(m_idx, n_idx), tile=accumulator)

def launch_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, transb=False, act=0, out_dtype=torch.bfloat16):
    stream = torch.cuda.current_stream()
    tile_m, tile_n, tile_k = 64, 64, 64
    M, N, K = a.shape[0], c.shape[1], a.shape[1]
    grid = (ceil(M/tile_m) * ceil(N/tile_n), 1, 1)
    
    ct.launch(stream,
            grid,  # 1D grid of processors
            matmul,
            (a, b, c, tile_m, tile_n, tile_k, transb, act))

# qwen2
#launch_matmul torch.Size([1, 8960]) torch.Size([1536, 8960])

def test():
    import time
    import torch
    # Create input data
    tile_m, tile_n, tile_k = 128, 64, 128
    M, N, K = 4096, 4096, 4096
    grid = (ceil(M/tile_m) * ceil(N/tile_n), 1, 1)
    print("Grid size:", grid)

    a = torch.rand((M, K), device='cuda', dtype=torch.bfloat16)
    b = torch.rand((K, N), device='cuda', dtype=torch.bfloat16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.bfloat16)

    # Launch kernel

    #benchmark this

    stream = torch.cuda.current_stream()
    args = (a, b, c, tile_m, tile_n, tile_k, False, 0)
    ct.launch(stream,
            grid,  # 1D grid of processors
            matmul,
            args)
    start = time.time()
    for _ in range(10):
        ct.launch(stream,
                grid,  # 1D grid of processors
                matmul,
                args)
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"Kernel execution time: {total:.4f} seconds")

    expected = torch.matmul(a, b)
    start = time.time()
    for _ in range(10):
        expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total = (time.time() - start)/10
    print(f"PyTorch matmul execution time: {total:.4f}) seconds")
    torch.testing.assert_close(c, expected)


    b = torch.rand((N, K), device='cuda', dtype=torch.bfloat16)  # transposed b
    ct.launch(stream,
            grid,  # 1D grid of processors
            matmul,
            (a, b, c, tile_m, tile_n, tile_k, True, 0))
    start = time.time()
    for _ in range(10):
        ct.launch(stream,
                grid,  # 1D grid of processors
                matmul,
                (a, b, c, tile_m, tile_n, tile_k, True, 0))
    torch.cuda.synchronize()
    total = (time.time() - start)/10


    expected = torch.matmul(a, b.T)
    start = time.time()
    for _ in range(10):
        expected = torch.matmul(a, b.T)
    torch.cuda.synchronize()
    end = time.time()
    total = (time.time() - start)/10
    print(f"Kernel execution time with transposed B: {total:.4f} seconds")
    torch.testing.assert_close(c, expected)

    print("✓ matmul_example passed!")


def test_gemv():
    import time
    import torch
    # Create input data
    tile_m, tile_n, tile_k = 16, 64, 128
    split_k = 16
    M, N, K = 1, 1536, 8960
    grid = (ceil(M/tile_m) * ceil(N/tile_n), split_k, 1)
    print("Grid size:", grid)

    a = torch.rand((M, K), device='cuda', dtype=torch.bfloat16)
    b = torch.rand((N, K), device='cuda', dtype=torch.bfloat16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.bfloat16)

    locks = torch.zeros((grid[0]), device='cuda', dtype=torch.int32)
    counts = torch.zeros((grid[0]), device='cuda', dtype=torch.int32)

    # Launch kernel

    #benchmark this

    stream = torch.cuda.current_stream()
    args = (a, b, c, locks, counts, tile_n, tile_k, split_k)
    ct.launch(stream,
            grid,  # 1D grid of processors
            gemv_split_k_kernel,
            args)
    start = time.time()
    for _ in range(10):
        ct.launch(stream,
                grid,  # 1D grid of processors
                gemv_split_k_kernel,
                args)
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"Kernel execution time: {total*1000:.4f} ms")

    expected = torch.matmul(a, b.T)
    start = time.time()
    for _ in range(10):
        expected = torch.matmul(a, b.T)
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"PyTorch matmul execution time: {total*1000:.4f} ms")
    torch.testing.assert_close(c, expected)
    print("✓ matmul_split_k passed!")


if __name__ == "__main__":
    # test()
    test_gemv()