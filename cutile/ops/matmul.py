
import time
import cuda.tile as ct
from math import ceil
import torch

ACT_RELU = 1
ACT_SILU = 2
ACT_NONE = 0

ConstInt = ct.Constant[int]

@ct.kernel(occupancy=ct.ByTarget(sm_120=2))
def gemv_split_k_kernel(A, B, C, f32acc, COUNTS,
                          tn: ConstInt, tk: ConstInt,
                          SPLIT_K: ConstInt):
    GROUP_SIZE_M = 1
    M = 1
    N = B.shape[1]
    bidx, bidy = 0, ct.bid(0)
    bidz = ct.bid(1)
    # pad tile A to fake_tm rows, to enable tensorcore
    fake_tm = 16
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(1, tk))
    sum = ct.full((fake_tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    split_size = ct.cdiv(num_tiles_k, SPLIT_K)
    for k in range(bidz * split_size, bidz * split_size + split_size, 1):
        # tile a has only one effective row, of shape tk. It is not efficient to use TMA to load it.
        a = ct.load(A, index=(bidx, k), shape=(fake_tm, tk),
                    padding_mode=zero_pad, allow_tma=False).astype(dtype)
        b = ct.load(B, index=(bidy, k), shape=(tn, tk),
                    padding_mode=zero_pad).astype(dtype)
        b = ct.transpose(b)
        sum = ct.mma(a, b, sum)
    # only the first row of sum is needed
    sum = ct.extract(sum, index=(0, 0), shape=(1, tn))
    sum = ct.reshape(sum, (tn,))
    count_offset = ct.bid(0)
    C_offset = ct.arange(tn, dtype=ct.int32) + bidy * tn
    ct.atomic_add(f32acc, (0, C_offset), sum)
    new_count = ct.atomic_add(COUNTS, count_offset, 1)
    if (new_count + 1) % SPLIT_K == 0:
        result = ct.gather(f32acc, (0, C_offset))
        ct.scatter(C, (0, C_offset), result.astype(C.dtype))
        ct.scatter(f32acc, (0, C_offset), 0)
        ct.scatter(COUNTS, count_offset, 0)

f32acc = None
counts = None
def launch_gemv(stream: torch.cuda.Stream, a: torch.Tensor, b1: torch.Tensor, c: torch.Tensor, tile_n=128, tile_k=128, split_k=16):
    M, N, K = a.shape[0], c.shape[1], a.shape[1]
    grid = (ceil(N/tile_n), split_k, 1)
    assert b1.shape == (N, K)
    assert M == 1

    global f32acc, counts
    if f32acc is None or f32acc.shape[1] < N:
        f32acc = torch.zeros((1, N), device='cuda', dtype=torch.float32)
    if counts is None or counts.shape[0] < grid[0]:
        counts = torch.zeros((grid[0],), device='cuda', dtype=torch.int32)
    args = (a, b1, c, f32acc, counts, tile_n, tile_k, split_k)
    ct.launch(stream,
            grid,
            gemv_split_k_kernel,
            args)



@ct.kernel(occupancy=ct.ByTarget(sm_120=2))
def matmul(a, b, c, TILE_M: ct.Constant[int], TILE_N: ct.Constant[int], TILE_K: ct.Constant[int], transb: ct.Constant[bool], act: ct.Constant[int], latencyAB: ct.Constant[int], latencyC: ct.Constant[int]):
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

    for k in range(num_tiles_k):
        a_tile = ct.load(a, index=(m_idx, k), shape=(TILE_M, TILE_K), padding_mode=padding_mode, latency=latencyAB)
        if transb:
            b_tile = ct.load(b, index=(n_idx, k), shape=(TILE_N, TILE_K), padding_mode=padding_mode, latency=latencyAB)
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
    ct.store(c, index=(m_idx, n_idx), tile=accumulator, latency=latencyC)

def launch_matmul(stream: torch.cuda.Stream, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, transb=False, act=0, tile_m=64, tile_n=64, tile_k=64, latencyAB=2, latencyC=2):
    M, N, K = a.shape[0], c.shape[1], a.shape[1]
    grid = (ceil(M/tile_m) * ceil(N/tile_n), 1, 1)
    
    ct.launch(stream,
            grid,  # 1D grid of processors
            matmul,
            (a, b, c, tile_m, tile_n, tile_k, transb, act, latencyAB, latencyC))

def launch_matmul_auto_tune(stream: torch.cuda.Stream, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, transb=False, act=0):
    M, N, K = a.shape[0], c.shape[1], a.shape[1]
    import cuda.tile_experimental as ct_experimental
    from types import SimpleNamespace
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (ceil(M/cfg.tile_m) * ceil(N/cfg.tile_n), 1, 1),
        kernel=matmul,
        args_fn=lambda cfg: (a, b, c, cfg.tile_m, cfg.tile_n, cfg.tile_k, transb, act, cfg.latencyAB, cfg.latencyC),
        hints_fn=lambda cfg: {
            "occupancy": cfg.occupancy,
        },
        search_space=[SimpleNamespace(tile_m=TM, tile_n=TN, tile_k=TK, occupancy=occupancy, latencyAB=latencyAB, latencyC=latencyC)
                for TM in [32, 64] for TN in [32, 64, 128] for TK in [32, 64, 128] for occupancy in [1, 2, 4, 8] for latencyAB in [1, 2, 4] for latencyC in [1, 2, 4]],
    )


@ct.kernel(occupancy=ct.ByTarget(sm_120=2))
def matmul_split_k_kernel(A, B, C, LOCKS, COUNTS,
                          tm: ConstInt, tn: ConstInt, tk: ConstInt,
                          SPLIT_K: ConstInt):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = C.shape[1]
    pid = ct.bid(0)
    num_tiles_n = ct.cdiv(N, tn)
    bidx, bidy = pid // num_tiles_n, pid % num_tiles_n
    bidz = ct.bid(1)

    num_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    sum = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    split_size = ct.cdiv(num_tiles_k, SPLIT_K)
    for k in range(bidz * split_size, bidz * split_size + split_size, 1):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                    padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(bidy, k), shape=(tn, tk),
                    padding_mode=zero_pad).astype(dtype)
        sum = ct.mma(a, b.transpose(), sum)

    sum = ct.astype(sum, C.dtype)
    lock_offset = ct.bid(0)
    count_offset = lock_offset
    while ct.atomic_cas(LOCKS, lock_offset, 0, 1, memory_order=ct.MemoryOrder.ACQUIRE) == 1:
        pass
    count = ct.gather(COUNTS, count_offset)
    if count == 0:
        ct.store(C, index=(bidx, bidy), tile=sum)
    else:
        curr = ct.load(C, index=(bidx, bidy), shape=(tm, tn))
        ct.store(C, index=(bidx, bidy), tile=(curr + sum))
    ct.scatter(COUNTS, count_offset, (count + 1) % SPLIT_K)
    ct.atomic_xchg(LOCKS, lock_offset, 0, memory_order=ct.MemoryOrder.RELEASE)


locks = None
def launch_gemm_split_k(stream: torch.cuda.Stream, a: torch.Tensor, b1: torch.Tensor, c: torch.Tensor):
    split_k, tile_m, tile_n, tile_k = 4, 64, 64, 64
    M, N, K = a.shape[0], c.shape[1], a.shape[1]
    grid = (ceil(N/tile_n)*ceil(M/tile_m), split_k, 1)
    assert b1.shape == (N, K)

    global locks, counts
    if locks is None or locks.numel() < grid[0]:
        locks = torch.zeros((grid[0],), device='cuda', dtype=torch.int32)
    if counts is None or counts.shape[0] < grid[0]:
        counts = torch.zeros((grid[0],), device='cuda', dtype=torch.int32)
    args = (a, b1, c, locks, counts, tile_m, tile_n, tile_k, split_k)
    ct.launch(stream,
            grid,
            matmul_split_k_kernel,
            args)



# qwen2
#launch_matmul torch.Size([1, 8960]) torch.Size([1536, 8960])

def bench_matmul(a, b, c, launch_func, iter=50, **kwargs):
    stream = torch.cuda.current_stream()
    import time
    launch_func(stream, a, b, c, **kwargs)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iter):
        launch_func(stream, a, b, c, **kwargs)
    torch.cuda.synchronize()
    total = (time.time() - start)/iter
    return total * 1000  # ms

def tune_matmul(a, b, c, cfgs, launch_func, **kwargs):
    best_time = float('inf')
    best_cfg = None
    for cfg in cfgs:
        total = bench_matmul(a, b, c, launch_func, iter=50, **(kwargs | cfg))
        print(f"matmul tune: {cfg}, time={total:.4f} ms")
        if total < best_time:
            best_time = total
            best_cfg = cfg
        time.sleep(0.5)
    print(f"Best matmul config: {best_cfg}, time={best_time:.4f} ms")

def test():
    import time
    import torch
    # Create input data
    tile_m, tile_n, tile_k, latencyAB, latencyC = 32, 64, 128, 1, 2
    M, N, K = 128, 1536, 8960
    grid = (ceil(M/tile_m) * ceil(N/tile_n), 1, 1)
    print("Grid size:", grid)

    a = torch.rand((M, K), device='cuda', dtype=torch.float16)
    b = torch.rand((K, N), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float16)

    # Launch kernel

    #benchmark this
    total = bench_matmul(a, b, c, launch_matmul, iter=50, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, latencyAB=latencyAB, latencyC=latencyC, act=0, transb=False)
    print(f"Kernel execution time: {total:.4f} ms")

    expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"PyTorch matmul execution time: {total*1000:.4f} ms")
    torch.testing.assert_close(c, expected)


    tile_m, tile_n, tile_k, latencyAB, latencyC  = 64, 32, 128, 2, 2
    b = torch.rand((N, K), device='cuda', dtype=torch.float16)  # transposed b
    total = bench_matmul(a, b, c, launch_matmul, iter=50, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, latencyAB=latencyAB, latencyC=latencyC, act=0, transb=True)
    print(f"Kernel execution time with transposed B: {total:.4f} ms")

    stream = torch.cuda.current_stream()
    c2 = torch.zeros((M, N), device='cuda', dtype=torch.float32)
    launch_gemm_split_k(stream, a, b, c2)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        launch_gemm_split_k(stream, a, b, c2)
    torch.cuda.synchronize()
    total = (time.time() - start)/10
    print(f"splitk execution time with transposed B: {total*1000:.4f} ms")

    expected = torch.matmul(a, b.T)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        expected = torch.matmul(a, b.T)
    torch.cuda.synchronize()
    end = time.time()
    total = (time.time() - start)/10
    print(f"Pytorch execution time with transposed B: {total*1000:.4f} ms")
    torch.testing.assert_close(c, expected)
    # torch.testing.assert_close(c2, expected)

    print("✓ matmul_example passed!")


def test_gemv():
    import time
    import torch
    # Create input data
    tile_m, tile_n, tile_k = 1, 128, 128
    split_k = 16
    M, N, K = 1, 1536, 8960
    grid = (ceil(N/tile_n), split_k, 1)
    print("Grid size:", grid)

    a = torch.rand((M, K), device='cuda', dtype=torch.float16)
    b = torch.rand((N, K), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device='cuda', dtype=torch.float16)

    f32acc = torch.zeros((M, N), device='cuda', dtype=torch.float32)
    counts = torch.zeros((grid[0]), device='cuda', dtype=torch.int32)

    # Launch kernel

    #benchmark this
    total = bench_matmul(a, b, c, launch_gemv, iter=50, tile_n=tile_n, tile_k=tile_k, split_k=split_k)
    print(f"Kernel execution time: {total:.4f} ms")


    expected = torch.matmul(a, b.T)
    torch.cuda.synchronize()
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
    # test_gemv()
    # tune_matmul(torch.rand((128, 8960), device='cuda', dtype=torch.float16),
    #             torch.rand((8960, 1536), device='cuda', dtype=torch.float16),
    #             torch.zeros((128, 1536), device='cuda', dtype=torch.float16),
    #             128, 1536, 8960, False)

    # tune matmul a @ b.T
    # cfgs = [{'tile_m': m, 'tile_n': n, 'tile_k': k, 'latencyAB': latencyAB, 'latencyC': latencyC} for m in [32,64,128] for n in [32,64,128] for k in [64,128] for latencyAB in [1,2] for latencyC in [1,2]]
    # tune_matmul(torch.rand((128, 8960), device='cuda', dtype=torch.float16),
    #             torch.rand((1536, 8960), device='cuda', dtype=torch.float16),
    #             torch.zeros((128, 1536), device='cuda', dtype=torch.float16), 
    #             cfgs, launch_matmul,
    #             act=0, transb=False)
    
    from autotune_logger import setup_autotune_logger
    setup_autotune_logger()
    launch_matmul_auto_tune(torch.cuda.current_stream(),
                            torch.rand((128, 8960), device='cuda', dtype=torch.float16),
                            torch.rand((1536, 8960), device='cuda', dtype=torch.float16),
                            torch.zeros((128, 1536), device='cuda', dtype=torch.float16),
                            transb=True, act=0)
    print("done")

    # tune gemv a @ b.T
    # cfgs = [{'tile_n': n, 'tile_k': k, 'split_k': splitk}  for n in [32,64,128] for k in [32,64,128] for splitk in [4,8,16]]
    # tune_matmul(torch.rand((1, 8960), device='cuda', dtype=torch.float16),
    #             torch.rand((1536, 8960), device='cuda', dtype=torch.float16),
    #             torch.zeros((1, 1536), device='cuda', dtype=torch.float16), 
    #             cfgs, launch_gemv)