import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr,       # pointer to predictions
    target_ptr,     # pointer to targets
    n_elements,     # total number of elements
    delta: tl.constexpr,  # threshold for Huber loss (default 1.0)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute differences
    diff = pred - target

    # Apply Huber loss: if |diff| <= delta, use linear loss; else, use quadratic
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff <= delta, 0.5 * diff * diff, abs_diff - 0.5 * delta)

    # Accumulate loss per element
    # We use a simple reduction over the block to sum up the loss
    # Note: We could accumulate in shared memory for larger blocks, but for simplicity and correctness,
    # we just return the per-element loss and let the outer loop sum it.
    # However, since we are not reducing across the entire batch in this kernel,
    # we return the per-element loss values for later summing in the host.

    # Store per-element loss
    tl.store(loss_ptr + offsets, loss, mask=mask)


@triton.jit
def smooth_l1_loss_kernel_sum(
    pred_ptr,       # pointer to predictions
    target_ptr,     # pointer to targets
    n_elements,     # total number of elements
    delta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel computes the total smooth L1 loss by reducing over the batch
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    diff = pred - target
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff <= delta, 0.5 * diff * diff, abs_diff - 0.5 * delta)

    # Reduce over the block to sum the loss
    # We use a simple reduction using shared memory or in-block reduction
    # Since we don't need per-element output, we just sum over the block
    # This is a reduction kernel that sums the loss per block
    # We use a simple loop to accumulate in a single value per block
    # For performance, we reduce across the block using a simple loop

    # We'll accumulate the loss in a single value per block
    # Use a temporary accumulator
    acc = tl.zeros(1, dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        if mask[i]:
            acc += loss[i]
    # Store the block sum in a temporary location (we will not store it back to global)
    # Instead, we rely on the outer loop to sum over all blocks
    # So we just compute the per-block sum and let the host sum it later
    # But we don't have a place to store it. So we modify the design.

    # Instead, we return the per-element loss and let the host sum it.
    # So we don't do reduction here — we just compute per-element loss.

    # We don't store the loss here; we just compute it.
    # So we modify the kernel to not return anything — we just compute loss per element
    # and the host will sum it up.
    # But we need to store it somewhere.

    # Alternative: we store the loss in a temporary buffer.
    # But we don't have a buffer in the kernel.

    # So we change approach: we compute the per-element loss and store it in a temporary buffer.
    # We create a temporary output pointer to store the loss per element.
    # We pass in a loss_ptr.

    # So we need to modify the kernel signature to include a loss_ptr.

    # We will refactor to include loss_ptr.
    pass


# Corrected version: kernel that computes per-element loss and stores it
@triton.jit
def smooth_l1_loss_kernel_with_output(
    pred_ptr,       # pointer to predictions
    target_ptr,     # pointer to targets
    loss_ptr,       # pointer to output loss per element
    n_elements,     # total number of elements
    delta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    diff = pred - target
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff <= delta, 0.5 * diff * diff, abs_diff - 0.5 * delta)

    tl.store(loss_ptr + offsets, loss, mask=mask)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0):
    """
    Compute Smooth L1 loss using a custom Triton kernel.
    
    Args:
        predictions: Tensor of shape (batch_size,)
        targets: Tensor of shape (batch_size,)
        delta: Threshold for Huber loss (default 1.0)

    Returns:
        Scalar loss value (sum of per-element losses)
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Ensure tensors are on the same device
    if predictions.device != targets.device:
        raise RuntimeError("Predictions and targets must be on the same device.")

    n_elements = predictions.numel()
    loss = torch.zeros(1, device=predictions.device, dtype=torch.float32)

    # Allocate temporary output tensor for per-element loss
    per_element_loss = torch.empty_like(predictions)

    # Launch kernel to compute per-element loss
    BLOCK_SIZE = 256  # Optimal for Ampere, power of 2
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Use autotuning to find best block size
    # For now, we use 256 as it's a good balance for memory and compute
    smooth_l1_loss_kernel_with_output[
        grid
    ](
        predictions.data_ptr(),
        targets.data_ptr(),
        per_element_loss.data_ptr(),
        n_elements,
        delta=delta,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Sum the per-element loss to get total loss
    total_loss = per_element_loss.sum().item()

    return total_loss


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)