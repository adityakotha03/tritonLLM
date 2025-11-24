import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mish_forward_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Softplus: log(1 + exp(x))
    # For stability, we use: x if x > threshold else log1p(exp(x))
    # Triton doesn't have log1p or exp directly, so we implement stable version
    # Approximate stable mish using tanh for performance
    # Mish(x) = x * tanh(softplus(x)) ≈ x * tanh(ln(1 + exp(x)))
    # We use the approximation: tanh(ln(1+exp(x))) = 2*sigmoid(x) - 1
    # But better to compute softplus with clamping
    exp_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # softplus(x)
    tanh_exp = tl.tanh(exp_x)
    out = x * tanh_exp

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mish_forward_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def addcdiv_kernel(
    input_ptr,
    tensor1_ptr,
    tensor2_ptr,
    value_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    tensor1 = tl.load(tensor1_ptr + offsets, mask=mask)
    tensor2 = tl.load(tensor2_ptr + offsets, mask=mask)
    value = tl.load(value_ptr)

    div = tensor1 / tensor2
    update = value * div
    output = input_val + update

    tl.store(out_ptr + offsets, output, mask=mask)


def triton_addcdiv(input_tensor, tensor1, tensor2, value):
    assert all(t.is_cuda for t in [input_tensor, tensor1, tensor2]), "All tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    tensor1 = tensor1.contiguous()
    tensor2 = tensor2.contiguous()
    out = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    value_tensor = torch.full((1,), value, dtype=input_tensor.dtype, device=input_tensor.device)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    addcdiv_kernel[grid](input_tensor, tensor1, tensor2, value_tensor, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for elementwise operations and fused Mish.
    The two subtractions are fused into a single addcdiv-like operation (but simpler: subtract constant).
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.register_buffer('total_sub', torch.tensor(subtract_value_1 + subtract_value_2))

    def forward(self, x):
        x = self.conv(x)
        # Fuse the two subtractions: x - a - b = x - (a + b)
        x = x - self.total_sub
        # Apply Mish using Triton kernel
        x = triton_mish(x)
        return x