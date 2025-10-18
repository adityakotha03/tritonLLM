import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 2}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048, "num_warps": 8}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096, "num_warps": 8}, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def relu_fwd_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.maximum(x, 0)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 2}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048, "num_warps": 8}, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096, "num_warps": 8}, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def relu_bwd_kernel(dy_ptr, y_ptr, dx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    grad = tl.where(y > 0, dy, 0)
    tl.store(dx_ptr + offsets, grad, mask=mask)


def _launch_relu_fwd(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Triton ReLU forward requires CUDA tensor"
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_fwd_kernel[grid](x_contig.view(-1), out.view(-1), n_elements)
    return out


def _launch_relu_bwd(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    assert y.is_cuda and dy.is_cuda, "Triton ReLU backward requires CUDA tensors"
    y_contig = y.contiguous()
    dy_contig = dy.contiguous()
    dx = torch.empty_like(dy_contig)
    n_elements = y_contig.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_bwd_kernel[grid](dy_contig.view(-1), y_contig.view(-1), dx.view(-1), n_elements)
    return dx


class _TritonReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            y = _launch_relu_fwd(x)
            ctx.save_for_backward(y)
            ctx.is_cuda_path = True
            return y
        else:
            y = torch.relu(x)
            ctx.save_for_backward(y)
            ctx.is_cuda_path = False
            return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (y,) = ctx.saved_tensors
        if ctx.is_cuda_path and grad_output.is_cuda:
            dx = _launch_relu_bwd(y, grad_output)
            return dx
        else:
            dx = grad_output * (y > 0).to(grad_output.dtype)
            return dx


class ModelNew(nn.Module):
    """
    Optimized model that performs a ReLU activation using a Triton kernel on CUDA.
    Falls back to torch.relu on CPU.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _TritonReLUFn.apply(x)


# Keep the same input helpers as the original for compatibility.
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []