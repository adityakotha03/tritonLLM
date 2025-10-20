import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_fwd_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_bwd_kernel(grad_out_ptr, ref_ptr, grad_in_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    go = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    ref = tl.load(ref_ptr + offsets, mask=mask, other=0.0)  # ref can be input or output from forward
    grad_in = tl.where(ref > 0, go, 0.0)
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


def _triton_relu_forward(x: torch.Tensor) -> torch.Tensor:
    x_ctg = x.contiguous()
    y = torch.empty_like(x_ctg)
    n_elements = x_ctg.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    relu_fwd_kernel[grid](x_ctg, y, n_elements)
    return y


def _triton_relu_backward(grad_out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    go = grad_out.contiguous()
    ref_ctg = ref.contiguous()
    grad_in = torch.empty_like(go)
    n_elements = go.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    relu_bwd_kernel[grid](go, ref_ctg, grad_in, n_elements)
    return grad_in


class _ReLUTritionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        if x.is_cuda:
            y = _triton_relu_forward(x)
            # Save output for mask in backward
            ctx.save_for_backward(y)
            ctx.use_triton = True
            return y
        else:
            y = torch.relu(x)
            ctx.save_for_backward(y)
            ctx.use_triton = False
            return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (ref,) = ctx.saved_tensors
        if ctx.use_triton and grad_output.is_cuda:
            grad_input = _triton_relu_backward(grad_output, ref)
            return grad_input
        else:
            grad_input = grad_output * (ref > 0).to(grad_output.dtype)
            return grad_input


class ModelNew(nn.Module):
    """
    Triton-optimized ReLU activation module with autograd support and CPU fallback.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ReLUTritionFn.apply(x)