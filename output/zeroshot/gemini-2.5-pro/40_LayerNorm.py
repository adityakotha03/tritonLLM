import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _layer_norm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,   # Pointers to tensors
    stride_x_row, stride_y_row,  # Strides for X and Y
    N,                           # Size of the normalization dimension
    eps,                         # Epsilon for stability
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for the forward pass of Layer Normalization.
    It processes one row of the input matrix per program instance.
    The kernel is fused: it computes mean/variance and then normalizes in a single pass.
    """
    # Each program instance handles one row of the input.
    row_idx = tl.program_id(0)

    # Pointers to the start of the current row for input and output.
    X_row_start_ptr = X_ptr + row_idx * stride_x_row
    Y_row_start_ptr = Y_ptr + row_idx * stride_y_row

    # --- Pass 1: Compute mean and variance ---
    # Accumulators for sum and sum of squares, using float32 for precision.
    mean_acc = 0.
    var_acc = 0.
    
    # Iterate over the row in blocks of size BLOCK_SIZE.
    for off in range(0, N, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # Load a block of data from the input tensor.
        # Use float32 for reduction to maintain precision.
        x = tl.load(X_row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Update accumulators for mean and variance calculation.
        mean_acc += tl.sum(x, axis=0)
        var_acc += tl.sum(x * x, axis=0)
        
    # Finalize mean and variance calculations.
    mean = mean_acc / N
    var = (var_acc / N) - (mean * mean)
    
    # Compute reciprocal of standard deviation.
    rstd = 1.0 / tl.sqrt(var + eps)

    # --- Pass 2: Normalize, scale, and shift ---
    # Iterate over the row again to apply the normalization.
    for off in range(0, N, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # Load the input block again.
        x = tl.load(X_row_start_ptr + offsets, mask=mask, other=0.0)
        
        # Load weight and bias.
        w = tl.load(W_ptr + offsets, mask=mask)
        b = tl.load(B_ptr + offsets, mask=mask)
        
        # Compute the normalized output.
        y_normalized = (x - mean) * rstd
        
        # Apply affine transformation (scale and shift).
        y_scaled_shifted = y_normalized * w + b
            
        # Store the final result.
        tl.store(Y_row_start_ptr + offsets, y_scaled_shifted, mask=mask)


class TritonLayerNorm(nn.Module):
    """
    A Layer Normalization module implemented with a custom Triton kernel.
    This implementation is a drop-in replacement for torch.nn.LayerNorm.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor):
        # Ensure input tensor shape matches the normalization shape at the end.
        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape, "Input shape mismatch"
        
        # Reshape the input tensor to 2D (M, N) for the kernel.
        # M is the number of rows (batch dimension), N is the size of the normalization dimension.
        original_shape = x.shape
        N = int(torch.prod(torch.tensor(self.normalized_shape)).item())
        M = x.numel() // N
        x_reshaped = x.reshape(M, N)
        
        # Ensure input is contiguous.
        x_reshaped = x_reshaped.contiguous()
        
        # Prepare weight and bias tensors. If no affine transform, use 1s and 0s.
        if self.elementwise_affine:
            weight = self.weight.contiguous()
            bias = self.bias.contiguous()
        else:
            weight = torch.ones(N, dtype=x.dtype, device=x.device)
            bias = torch.zeros(N, dtype=x.dtype, device=x.device)

        # Create an empty output tensor.
        y = torch.empty_like(x_reshaped)

        # The grid is 1D, with one program instance per row of the input matrix.
        grid = (M,)

        # Launch the Triton kernel.
        _layer_norm_fwd_kernel[grid](
            x_reshaped,
            y,
            weight,
            bias,
            x_reshaped.stride(0),
            y.stride(0),
            N,
            self.eps,
            # BLOCK_SIZE is determined by the autotuner.
        )
        
        # Reshape the output back to the original shape.
        return y.reshape(original_shape)


class ModelNew(nn.Module):
    """
    Simple model that performs Layer Normalization using a custom Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the TritonLayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        # Replace nn.LayerNorm with our custom TritonLayerNorm
        self.ln = TritonLayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)
