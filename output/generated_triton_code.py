import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    x_ptr,  # *const T: input
    y_ptr,  # *mut T: output
    n_elements,  # int32 / int64: total elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Use max(x, 0) with safe dtype handling
    zero = x * 0
    y = tl.maximum(x, zero)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "triton_relu requires a CUDA tensor"
    # Only support floating point tensors; otherwise, fallback is used by caller.
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64), "Unsupported dtype for triton_relu"

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    relu_kernel[grid](
        x_contig, out, n_elements,
    )
    # If input was non-contiguous, match original layout by reshaping
    return out.view_as(x)


class ModelNew(nn.Module):
    """
    Optimized model that performs ReLU activation using a custom Triton kernel on CUDA tensors.
    Falls back to torch.relu on CPU or unsupported dtypes.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return triton_relu(x)
        else:
            return torch.relu(x)


# Match the original interface
batch_size = 16
dim = 16384

def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []