import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    in_out_ptr0,  # Pointer to input/output tensor
    xnumel,  # Total number of elements
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([1], xnumel, tl.int64)
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tmp0 <= 0
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tl.where(tmp1, tmp2, tmp0)
    tl.store(in_out_ptr0 + x0, tmp3, xmask)


def triton_relu(input_0):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the input tensor is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input_0.is_cuda, "Tensor must be on CUDA."
    input_0 = input_0.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input_0)

    # Number of elements in the tensor
    xnumel = input_0.numel()
    XBLOCK = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((xnumel + meta["XBLOCK"] - 1) // meta["XBLOCK"],)

    # Launch the Triton kernel
    relu_kernel[grid](output, xnumel, XBLOCK=XBLOCK)
    return output


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return triton_relu(x)