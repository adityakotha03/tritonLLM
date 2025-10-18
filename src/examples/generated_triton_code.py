import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n_elements: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors."
    assert a.shape == b.shape, "Input tensors must have the same shape."
    assert a.dtype == b.dtype, "Input tensors must have the same dtype."
    # Ensure contiguous memory for coalesced accesses
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)

    n_elements = a.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, out, n_elements)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return triton_add(a, b)


def get_inputs():
    a = torch.randn(1, 128, device="cuda", dtype=torch.float32)
    b = torch.randn(1, 128, device="cuda", dtype=torch.float32)
    return [a, b]


def get_init_inputs():
    return []