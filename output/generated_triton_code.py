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
    key=["N"],
)
@triton.jit
def add_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0)
    b = tl.load(B_ptr + offs, mask=mask, other=0)
    c = a + b
    tl.store(C_ptr + offs, c, mask=mask)


class _TritonAddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        if not (a.is_cuda and b.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors.")
        if a.shape != b.shape:
            raise RuntimeError(f"Shape mismatch: {a.shape} vs {b.shape}")
        if a.dtype != b.dtype:
            raise RuntimeError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")

        a_c = a.contiguous()
        b_c = b.contiguous()
        N = a_c.numel()
        c = torch.empty_like(a_c)

        if N > 0:
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
            add_kernel[grid](a_c, b_c, c, N)

        # Save nothing for backward; gradients are passthrough
        return c.view_as(a)

    @staticmethod
    def backward(ctx, grad_out):
        # d(a+b)/da = 1, d(a+b)/db = 1
        return grad_out, grad_out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return _TritonAddFn.apply(a, b)


def get_inputs():
    a = torch.randn(1, 128, device="cuda")
    b = torch.randn(1, 128, device="cuda")
    return [a, b]


def get_init_inputs():
    return []