import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    delta = predictions - targets
    abs_delta = tl.abs(delta)
    abs_delta = tl.where(abs_delta < 1.0, abs_delta, 1.0)
    output = 0.5 * abs_delta * abs_delta

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_smooth_l1(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    output = torch.empty_like(predictions)
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable for optimal performance

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    smooth_l1_kernel[grid](predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1(predictions, targets)