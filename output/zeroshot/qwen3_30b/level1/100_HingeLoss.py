import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute: 1 - predictions * targets
    inner_product = predictions * targets
    clamped = tl.clamp(1.0 - inner_product, min=0.0)

    # Write result to output
    tl.store(output_ptr + offsets, clamped, mask=mask)


def triton_hinge_loss(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    output = torch.empty(1, dtype=predictions.dtype, device=predictions.device)
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    hinge_loss_kernel[grid](predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)