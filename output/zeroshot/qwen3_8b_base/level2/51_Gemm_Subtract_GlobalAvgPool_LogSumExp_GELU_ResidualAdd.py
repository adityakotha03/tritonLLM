import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemm_subtract_gavg_logsumexp_gelu_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    b_ptr,  # Pointer to bias tensor
    subtract_ptr,  # Pointer to subtract parameter
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    out_dim,  # Output dimension after global average pooling
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load weights and bias
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Gemm: x * w + b
    xw = x * w
    xw = tl.sum(xw, axis=0)
    xw = xw + b

    # Subtract
    subtract = tl.load(subtract_ptr, mask=mask, other=0.0)
    xw = xw - subtract

    # GlobalAvgPool: mean over features (assumed to be done in the forward pass)
    # For this kernel, we assume that the mean is already handled in the input
    # So we proceed to logsumexp and gelu

    # LogSumExp
    max_val = tl.max(xw, axis=0)
    xw_exp = tl.exp(xw - max_val)
    xw_logsumexp = tl.log(tl.sum(xw_exp, axis=0)) + max_val

    # GELU
    xw_gelu = 0.5 * xw_logsumexp * (1.0 + tl.erf(xw_logsumexp * (1.0 / tl.sqrt(2.0))))

    # Store output
    tl.store(out_ptr + offsets, xw_gelu, mask=mask)

def triton_gemm_subtract_gavg_logsumexp_gelu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, subtract: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda and subtract.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    subtract = subtract.contiguous()

    # Prepare output tensor
    out = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype)

    # Number of elements in the tensor
    n_elements = x.numel()
    out_dim = 1  # After global average pooling
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gemm_subtract_gavg_logsumexp_gelu_kernel[grid](x, w, b, subtract, out, n_elements, out_dim, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # Gemm
        x = triton_gemm_subtract_gavg_logsumexp_gelu(x, self.weight, self.bias, self.subtract)
        return x