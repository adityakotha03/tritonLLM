import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_min_subtract_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_rows,
    n_cols,
    n_features,
    constant,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread block indices
    row = tl.program_id(0)  # row of output matrix
    col = tl.program_id(1)  # col of output matrix

    # Block size for matrix multiplication (tile size)
    block_size = BLOCK_SIZE

    # Compute the row and col offsets
    row_offset = row * block_size
    col_offset = col * block_size

    # Create row and column indices for the current tile
    row_indices = row_offset + tl.arange(0, block_size)
    col_indices = col_offset + tl.arange(0, block_size)

    # Mask to prevent out-of-bounds access
    row_mask = row_indices < n_rows
    col_mask = col_indices < n_cols

    # Load input matrix x (batch, n_features)
    x = tl.load(x_ptr + row_indices[:, None] * n_features + tl.arange(0, n_features)[None, :], 
                mask=row_mask[:, None] & (tl.arange(0, n_features)[None, :] < n_features), 
                other=0.0)

    # Load weights (n_features, n_cols)
    w = tl.load(w_ptr + tl.arange(0, n_features)[:, None] * n_cols + col_indices[None, :], 
                mask=(tl.arange(0, n_features)[:, None] < n_features) & col_mask[None, :], 
                other=0.0)

    # Perform matrix multiplication: x @ w
    # Use tensor cores with bfloat16 for high throughput
    acc = tl.dot(x, w, out_dtype=tl.float32)
    acc = acc.to(tl.float32)

    # Apply bias if available
    if b_ptr is not None:
        bias = tl.load(b_ptr + col_indices, mask=col_mask, other=0.0)
        acc = acc + bias[None, :]

    # Apply min with constant: min(acc, constant)
    acc = tl.minimum(acc, constant)

    # Subtract constant: acc - constant
    acc = acc - constant

    # Store output
    tl.store(out_ptr + row_indices[:, None] * n_cols + col_indices[None, :], 
             acc, 
             mask=row_mask[:, None] & col_mask[None, :])


def triton_linear_min_subtract(x, w, b, constant):
    """
    Perform a fused linear + min + subtraction using Triton kernel.
    """
    assert x.is_cuda and w.is_cuda and (b is None or b.is_cuda), "All tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous() if b is not None else None

    # Get dimensions
    n_rows, n_features = x.shape
    n_cols = w.shape[1]

    # Output tensor
    out = torch.empty(n_rows, n_cols, dtype=x.dtype, device=x.device)

    # Choose block size (power of 2, 256 is good for A100)
    BLOCK_SIZE = 256

    # Grid dimensions: (n_rows // block_size, n_cols // block_size)
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]), triton.cdiv(n_cols, meta["BLOCK_SIZE"]))

    # Launch kernel
    linear_min_subtract_kernel[grid](
        x, w, b, out, n_rows, n_cols, n_features, constant, BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant, dtype=torch.float32))

    def forward(self, x):
        # Fuse linear, min, and subtraction into one Triton kernel
        return triton_linear_min_subtract(x, self.linear.weight, self.linear.bias, self.constant)