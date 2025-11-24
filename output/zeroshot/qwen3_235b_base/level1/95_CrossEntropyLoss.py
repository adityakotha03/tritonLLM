import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(
    input_ptr,  # Pointer to input logits (B, C)
    output_ptr,  # Pointer to output logsumexp (B,)
    B,  # Batch size
    C,  # Number of classes
    stride_input_b,  # Stride for batch in input
    BLOCK_SIZE_C: tl.constexpr,
):
    # One program per batch element
    batch_idx = tl.program_id(0)
    # Compute offset for this batch
    input_batch_offset = batch_idx * stride_input_b
    input_ptrs = input_ptr + input_batch_offset + tl.arange(0, BLOCK_SIZE_C)

    # Load all logits for this batch element
    mask = tl.arange(0, BLOCK_SIZE_C) < C
    logits = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Subtract max for numerical stability (online max reduction)
    max_logits = tl.max(logits, axis=0)
    logits_minus_max = logits - max_logits
    exp_logits = tl.exp(logits_minus_max)

    # Sum exponentials
    sum_exp = tl.sum(exp_logits, axis=0)

    # Compute log(sum(exp)) + max
    logsumexp = tl.log(sum_exp) + max_logits

    # Store result
    output_offsets = batch_idx
    tl.store(output_ptr + output_offsets, logsumexp)


@triton.jit
def cross_entropy_kernel(
    predictions_ptr,  # Pointer to predictions (logits) (B, C)
    targets_ptr,  # Pointer to targets (B,)
    output_ptr,  # Pointer to output loss (scalar)
    logsumexp_ptr,  # Pointer to precomputed logsumexp (B,)
    B,  # Batch size
    C,  # Number of classes
    stride_pred_b,  # Stride for batch in predictions
    reduction: tl.constexpr,  # 0 = none, 1 = mean, 2 = sum
    BLOCK_SIZE_B: tl.constexpr,
):
    # Each block handles up to BLOCK_SIZE_B elements
    b_start = tl.program_id(0) * BLOCK_SIZE_B
    b_range = tl.arange(0, BLOCK_SIZE_B)
    b_offsets = b_start + b_range
    mask = b_offsets < B

    # Load targets
    targets = tl.load(targets_ptr + b_offsets, mask=mask, other=0)
    # Load logsumexp values
    lse = tl.load(logsumexp_ptr + b_offsets, mask=mask, other=0.0)

    # Compute offset for predictions
    pred_offsets = b_offsets * stride_pred_b + targets
    pred_ptrs = predictions_ptr + pred_offsets
    correct_logits = tl.load(pred_ptrs, mask=mask, other=0.0)

    # Compute per-sample loss: logsumexp - correct_logit
    loss = lse - correct_logits

    # Store per-sample loss if needed, or reduce
    if reduction == 0:
        tl.store(output_ptr + b_offsets, loss, mask=mask)
    else:
        # Block-local reduction
        sum_loss = tl.sum(loss, axis=0)
        # Use atomics for global reduction
        if reduction == 1:
            # Mean: we'll divide later
            tl.atomic_add(output_ptr, sum_loss)
        elif reduction == 2:
            tl.atomic_add(output_ptr, sum_loss)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean'):
    B, C = predictions.shape
    assert targets.shape == (B,), f"Targets shape {targets.shape}, expected ({B},)"

    # Step 1: Compute logsumexp of logits (B,)
    logsumexp = torch.empty((B,), dtype=torch.float32, device=predictions.device)
    BLOCK_SIZE_C = triton.next_power_of_2(C)

    # Use small grid for logsumexp (one block per row)
    grid_logsumexp = (B,)
    logsumexp_kernel[grid_logsumexp](
        predictions,
        logsumexp,
        B, C,
        predictions.stride(0),
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )

    # Step 2: Compute final loss
    if reduction == 'none':
        output = torch.empty((B,), dtype=torch.float32, device=predictions.device)
        grid_loss = (triton.cdiv(B, 128),)
        cross_entropy_kernel[grid_loss](
            predictions,
            targets,
            output,
            logsumexp,
            B, C,
            predictions.stride(0),
            reduction=0,
            BLOCK_SIZE_B=128,
        )
        return output

    else:
        # Initialize output as zero scalar
        output = torch.zeros((), dtype=torch.float32, device=predictions.device)
        grid_loss = (triton.cdiv(B, 128),)
        cross_entropy_kernel[grid_loss](
            predictions,
            targets,
            output,
            logsumexp,
            B, C,
            predictions.stride(0),
            reduction=1 if reduction == 'mean' else 2,
            BLOCK_SIZE_B=128,
        )

        if reduction == 'mean':
            output = output / B
        return output


class ModelNew(nn.Module):
    """
    Optimized version of cross entropy loss using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets, reduction='mean')