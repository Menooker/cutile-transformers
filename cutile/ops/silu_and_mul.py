# modified from https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/silu_and_mul.py
import cuda.tile as ct
import torch
from cuda.tile._numeric_semantics import RoundingMode as RMd

ConstInt = ct.Constant[int]
def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1

@ct.kernel
def silu_and_mul(
    input,
    output,
    TILE_SIZE: ConstInt,
    hidden_size,
    approx: ct.Constant[bool],
):
    bid = ct.bid(0)  # this gives us our row
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)

    # For 2D input (batch_size, 2*hidden_size), we need 2D indices
    # Row index is just bid (scalar), column indices are offsets-based
    row_idx = bid
    a_col_idx = offsets  # First half: [0, hidden_size)
    b_col_idx = offsets + hidden_size  # Second half: [hidden_size, 2*hidden_size)

    # Load tiles using gather with 2D indices
    # gather broadcasts (scalar, tile) to (tile,)
    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True).astype(torch.float32)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True).astype(torch.float32)

    # Implement sigmoid for SiLU
    denom = ct.add(1, ct.exp(-a_tile), flush_to_zero=True)
    rounding_mode = RMd.APPROX if approx else None
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=rounding_mode)

    # Perform SiLU(a) * b
    silu_a = ct.mul(a_tile, sigmoid_a, flush_to_zero=True)
    result = ct.mul(silu_a, b_tile, flush_to_zero=True)
    result = ct.astype(result, input.dtype)

    # Store result using scatter with 2D indices
    # output is also 2D: (batch_size, hidden_size)
    out_col_idx = offsets
    ct.scatter(output, (row_idx, out_col_idx), result, check_bounds=True)

def launch_silu_and_mul(input: torch.Tensor, output: torch.Tensor, approx: bool = True):
    """
    input: (batch_size, 2*hidden_size)
    output: (batch_size, hidden_size)
    """
    stream = torch.cuda.current_stream()
    batch_size, total_hidden_size = input.shape
    hidden_size = total_hidden_size // 2
    grid = (batch_size, 1, 1)
    assert output.shape == (batch_size, hidden_size)
    ct.launch(
        stream,
        grid,
        silu_and_mul,
        (input, output, next_power_of_two(hidden_size), hidden_size, approx),
    )

if __name__ == "__main__":
    import torch
    # Simple test
    batch_size = 8
    hidden_size = 128
    input = torch.rand((batch_size, 2 * hidden_size), device='cuda', dtype=torch.float16)
    output = torch.zeros((batch_size, hidden_size), device='cuda', dtype=torch.float16)

    launch_silu_and_mul(input, output)

    # Verify results
    input_fp32 = input.float()
    a = input_fp32[:, :hidden_size]
    b = input_fp32[:, hidden_size:]
    expected = a * torch.sigmoid(a) * b
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)

    print("✓ silu_and_mul test passed!")