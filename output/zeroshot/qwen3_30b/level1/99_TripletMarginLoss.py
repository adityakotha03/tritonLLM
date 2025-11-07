import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    loss_ptr,
    n_elements,
    batch_size: tl.constexpr,
    dim: tl.constexpr,
    margin: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block offset
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load anchor, positive, negative vectors (row-wise, batch-dim)
    anchor = tl.load(anchor_ptr + offsets[:, None], mask=mask[:, None], other=0.0)
    positive = tl.load(positive_ptr + offsets[:, None], mask=mask[:, None], other=0.0)
    negative = tl.load(negative_ptr + offsets[:, None], mask=mask[:, None], other=0.0)

    # Compute squared distances: ||a - p||², ||a - n||²
    # Using dot product for squared L2: ||x - y||² = ||x||² + ||y||² - 2*x·y
    anchor_sq = tl.sum(anchor * anchor, axis=1)
    positive_sq = tl.sum(positive * positive, axis=1)
    negative_sq = tl.sum(negative * negative, axis=1)

    dot_ap = tl.sum(anchor * positive, axis=1)
    dot_an = tl.sum(anchor * negative, axis=1)

    dist_ap = anchor_sq + positive_sq - 2 * dot_ap
    dist_an = anchor_sq + negative_sq - 2 * dot_an

    # Compute triplet loss: max(0, dist_ap - dist_an + margin)
    loss = tl.maximum(0.0, dist_ap - dist_an + margin)

    # Store per-sample loss (reduce across dim)
    tl.store(loss_ptr + offsets, loss, mask=mask)


@triton.jit
def reduce_sum_kernel(
    loss_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Reduce sum across batch dimension
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    loss = tl.load(loss_ptr + offsets, mask=mask, other=0.0)
    sum_loss = tl.sum(loss, axis=0)

    # Store result (only one value)
    tl.store(output_ptr, sum_loss)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Ensure contiguous tensors
        anchor = anchor.contiguous()
        positive = positive.contiguous()
        negative = negative.contiguous()

        # Get total number of elements in batch
        batch_size = anchor.shape[0]
        n_elements = batch_size

        # Output tensor
        loss = torch.empty(1, device=anchor.device, dtype=anchor.dtype)

        # Triton kernel: compute triplet loss per sample
        BLOCK_SIZE = 1024  # Tuned for A100, coalesced access, good occupancy
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        triplet_margin_loss_kernel[grid](
            anchor, positive, negative, loss, n_elements, batch_size, anchor.shape[1], self.margin, BLOCK_SIZE=BLOCK_SIZE
        )

        # Final reduction: sum over batch
        output = torch.empty(1, device=anchor.device, dtype=anchor.dtype)
        reduce_sum_kernel[grid](loss, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        # Average over batch
        return output / batch_size