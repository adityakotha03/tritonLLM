import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triton_mul(x_ptr, scalar_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    xnumel = n_elements
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([1], xnumel, tl.int64)
    mask = xindex < xnumel
    x = tl.load(x_ptr + xindex, mask=mask, other=0.0)
    scalar = tl.load(scalar_ptr, mask=mask, other=0.0)
    tl.store(out_ptr + xindex, x * scalar, mask=mask)


def triton_mul_new(x: torch.Tensor, scalar: float):
    assert x.is_cuda
    scalar_tensor = torch.tensor(scalar, device=x.device)
    assert scalar_tensor.is_cuda
    x = x.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    triton_mul[grid](x, scalar_tensor, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_mul_new(A, s)