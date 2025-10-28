import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    ],
    key=["cols"],
)
@triton.jit
def _cumprod_pass1_scan_tiles(
    x_ptr,
    y_ptr,
    block_prod_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    num_tiles,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_tile = tl.program_id(1)
    # compute tile start
    col_start = pid_tile * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < cols

    # row-major 2D indexing
    row_off = pid_row * stride_row
    x = tl.load(x_ptr + row_off + offs * stride_col, mask=mask, other=1)
    # Ensure OOB lanes are neutral for product
    x = tl.where(mask, x, 1)

    idx = tl.arange(0, BLOCK_SIZE)
    # Inclusive scan (product) via iterative doubling
    shift = 1
    one = 1
    while shift < BLOCK_SIZE:
        src_idx = tl.maximum(idx - shift, 0)
        y = tl.where(idx >= shift, x[src_idx], one)
        x = x * y
        shift *= 2

    # Store scanned tile to output
    tl.store(y_ptr + row_off + offs * stride_col, x, mask=mask)

    # Compute tile product = last valid element of scanned tile
    n_rem = cols - col_start
    # valid elements in this tile
    valid = tl.minimum(n_rem, BLOCK_SIZE)
    last_idx = valid - 1
    # pick the last element robustly
    is_last = idx == last_idx
    # sum over lanes to extract the last element
    last_val = tl.sum(tl.where(is_last, x, 0), axis=0)
    # store per-tile product
    tl.store(block_prod_ptr + pid_row * num_tiles + pid_tile, last_val)


@triton.jit
def _cumprod_pass2_scan_block_prods(
    block_prod_ptr,
    carry_ptr,
    rows,
    num_tiles,
    BLOCK_TILES: tl.constexpr,
):
    pid_row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < num_tiles

    base = pid_row * num_tiles
    vals = tl.load(block_prod_ptr + base + offs, mask=mask, other=1)

    idx = tl.arange(0, BLOCK_TILES)
    one = 1

    # Build exclusive input: [1, vals[0], vals[1], ...]
    prev_idx = tl.maximum(idx - 1, 0)
    prev_vals = vals[prev_idx]
    ex = tl.where(idx == 0, one, tl.where(idx < num_tiles, prev_vals, one))

    # Inclusive scan on ex to get exclusive prefix products
    shift = 1
    while shift < BLOCK_TILES:
        src_idx = tl.maximum(idx - shift, 0)
        y = tl.where(idx >= shift, ex[src_idx], one)
        ex = ex * y
        shift *= 2

    tl.store(carry_ptr + base + offs, ex, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    ],
    key=["cols"],
)
@triton.jit
def _cumprod_pass3_apply_carry(
    y_ptr,
    carry_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    num_tiles,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_tile = tl.program_id(1)
    col_start = pid_tile * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < cols

    row_off = pid_row * stride_row
    carry = tl.load(carry_ptr + pid_row * num_tiles + pid_tile)
    y = tl.load(y_ptr + row_off + offs * stride_col, mask=mask, other=1)
    y = y * carry
    tl.store(y_ptr + row_off + offs * stride_col, y, mask=mask)


def _next_pow2_leq(max_val, cap=1024):
    v = 1
    while v < max_val and v < cap:
        v *= 2
    return v


def triton_cumprod(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input must be a CUDA tensor for Triton cumprod."
    if x.numel() == 0:
        return x.clone()

    ndim = x.ndim
    dim = dim if dim >= 0 else dim + ndim
    assert 0 <= dim < ndim, "Invalid dim."

    # Move scan dimension to the last and make contiguous
    perm = [d for d in range(ndim) if d != dim] + [dim]
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    x_perm = x.permute(perm).contiguous()
    rows = int(torch.tensor(x_perm.shape[:-1]).prod().item()) if x_perm.ndim > 1 else 1
    cols = x_perm.shape[-1]
    x_2d = x_perm.reshape(rows, cols)

    y_2d = torch.empty_like(x_2d)

    # Strides for 2D row-major (last dim contiguous)
    stride_row = cols
    stride_col = 1

    # Number of tiles along columns
    # BLOCK_SIZE picked by autotuner; grid computes num tiles using meta in kernels, but we still need tensor allocations
    # so compute with a conservative default BLOCK_SIZE=1024 for allocation.
    default_block = 1024
    num_tiles = (cols + default_block - 1) // default_block
    if num_tiles == 0:
        num_tiles = 1

    block_prod = torch.empty((rows, num_tiles), dtype=x.dtype, device=x.device)
    carry = torch.empty_like(block_prod)

    # Launch pass 1: per-tile inclusive scans and per-tile products
    grid1 = lambda meta: (rows, triton.cdiv(cols, meta["BLOCK_SIZE"]))
    _cumprod_pass1_scan_tiles[grid1](
        x_2d,
        y_2d,
        block_prod,
        rows,
        cols,
        stride_row,
        stride_col,
        triton.cdiv(cols, 1 if "BLOCK_SIZE" not in {} else 1),  # not used inside, kept for signature consistency
        cols=cols,
    )

    # Recompute actual num_tiles based on chosen BLOCK_SIZE in pass1's meta is not accessible here.
    # For correctness, recompute with the largest config (1024) to over-allocate; we already did.
    # For pass2, set BLOCK_TILES to next power-of-two >= num_tiles (capped to 1024)
    num_tiles_actual = (cols + default_block - 1) // default_block
    if num_tiles_actual == 0:
        num_tiles_actual = 1
    BLOCK_TILES = _next_pow2_leq(num_tiles_actual, cap=1024)

    # Launch pass 2: exclusive prefix products over tile products => carry per tile
    grid2 = (rows,)
    _cumprod_pass2_scan_block_prods[grid2](
        block_prod,
        carry,
        rows,
        num_tiles_actual,
        BLOCK_TILES=BLOCK_TILES,
    )

    # Launch pass 3: apply carry to scanned tiles
    grid3 = lambda meta: (rows, triton.cdiv(cols, meta["BLOCK_SIZE"]))
    _cumprod_pass3_apply_carry[grid3](
        y_2d,
        carry,
        rows,
        cols,
        stride_row,
        stride_col,
        triton.cdiv(cols, 1 if "BLOCK_SIZE" not in {} else 1),  # not used inside, kept for signature consistency
        cols=cols,
    )

    # Reshape back and inverse permute
    y_perm = y_2d.reshape(x_perm.shape)
    y = y_perm.permute(inv_perm)
    return y


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        if not x.is_cuda:
            # Fallback to PyTorch implementation on CPU
            return torch.cumprod(x, dim=self.dim)
        return triton_cumprod(x, self.dim)