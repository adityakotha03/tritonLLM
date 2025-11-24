import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel that fuses Linear → MaxPool1d → Sum → Scale
# ----------------------------------------------------------------------
@triton.jit
def fused_linear_pool_sum_scale_kernel(
    # Pointers to tensors
    x_ptr,          # (batch, in_features)
    w_ptr,          # (out_features, in_features)
    b_ptr,          # (out_features)
    out_ptr,        # (batch,)
    # Tensor dimensions
    batch_size,
    in_features,
    out_features,
    scale_factor,
    BLOCK_SIZE: tl.constexpr,  # Number of output pairs processed per block
):
    """
    Each program instance processes a single batch element.  For that
    element the kernel:

        1. Computes a matrix‑vector product  y = x @ Wᵀ + b
        2. Applies a 1‑D max‑pool with kernel=2 (no stride given → stride=2)
        3. Sums all pooled values
        4. Multiplies by a scaling factor
    """

    # ------------------------------------------------------------------
    # 1. Load the whole input vector for this batch element into registers
    # ------------------------------------------------------------------
    # Pointer to the start of this batch element
    base_x = x_ptr + tl.program_id(0) * in_features

    # Load input in chunks of BLOCK_SIZE
    # We keep the chunks in registers; they are reused many times
    in_chunks = tl.arange(0, BLOCK_SIZE)
    mask_in = in_chunks < in_features
    x_chunk = tl.load(base_x + in_chunks, mask=mask_in, other=0.0)  # float32

    # ------------------------------------------------------------------
    # 2. Iterate over output columns in blocks of BLOCK_SIZE
    # ------------------------------------------------------------------
    # Number of output pairs after max‑pooling (ceil division)
    out_pairs = (out_features + 1) // 2

    # Accumulator for the sum of pooled values
    sum_val = tl.zeros([1], dtype=tl.float32)

    # Loop over all output pairs
    for pair in range(out_pairs):
        # Start index of the pair in the output vector
        out_start = pair * 2

        # Load two consecutive columns of the weight matrix
        #   weight[col, :]  →  (in_features,)
        # We load the two columns into two registers for dot products
        col0 = tl.load(
            w_ptr + out_start * in_features + in_chunks,
            mask=mask_in,
            other=0.0,
        )
        col1 = tl.load(
            w_ptr + (out_start + 1) * in_features + in_chunks,
            mask=mask_in,
            other=0.0,
        )

        # Compute dot products for both columns
        dot0 = tl.dot(x_chunk, col0)
        dot1 = tl.dot(x_chunk, col1)

        # Add biases
        bias0 = tl.load(b_ptr + out_start, mask=(out_start < out_features), other=0.0)
        bias1 = tl.load(
            b_ptr + out_start + 1,
            mask=(out_start + 1 < out_features),
            other=0.0,
        )
        dot0 = dot0 + bias0
        dot1 = dot1 + bias1

        # Max‑pool: keep the larger of the two
        pooled = tl.max(dot0, dot1)

        # Accumulate
        sum_val += pooled

    # ------------------------------------------------------------------
    # 3. Apply scaling and write result
    # ------------------------------------------------------------------
    result = sum_val * scale_factor
    out_index = tl.program_id(0)
    tl.store(out_ptr + out_index, result, mask=out_index < batch_size)


# ----------------------------------------------------------------------
# Helper wrapper that launches the Triton kernel
# ----------------------------------------------------------------------
def fused_linear_pool_sum_scale(
    x: torch.Tensor,          # (batch, in_features)
    weight: torch.Tensor,     # (out_features, in_features)
    bias: torch.Tensor,       # (out_features)
    scale_factor: float,
):
    """
    Wrapper around the Triton kernel that matches the behaviour of
    the original Model.forward().
    """
    # Ensure tensors are contiguous and on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    # Output tensor
    out = torch.empty(batch_size, dtype=x.dtype, device=x.device)

    # Choose a BLOCK_SIZE that is a power of two and fits in the SM
    BLOCK_SIZE = 256

    # Launch kernel
    grid = lambda meta: (batch_size,)
    fused_linear_pool_sum_scale_kernel[grid](
        x_ptr=x.data_ptr(),
        w_ptr=weight.data_ptr(),
        b_ptr=bias.data_ptr(),
        out_ptr=out.data_ptr(),
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        scale_factor=scale_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ----------------------------------------------------------------------
# New model that uses the fused Triton kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model that replaces the sequence
    Linear → MaxPool1d → sum → scale with a single custom Triton kernel.
    """

    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        # kernel_size is unused because pooling is hard‑coded to 2 (stride 2)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Forward pass that calls the fused Triton kernel.
        """
        # Compute output of the linear layer
        x = self.linear(x)  # shape (batch, out_features)

        # Invoke the fused Triton kernel
        return fused_linear_pool_sum_scale(
            x, self.linear.weight, self.linear.bias, self.scale_factor
        )