import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,      # pointer to anchor embeddings
    positive_ptr,    # pointer to positive embeddings
    negative_ptr,    # pointer to negative embeddings
    batch_size,      # total batch size
    dim,             # dimension of embeddings
    margin: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    # Load anchor, positive, and negative embeddings
    anchor = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
    positive = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
    negative = tl.load(negative_ptr + offsets, mask=mask, other=0.0)

    # Compute distances: ||anchor - positive|| and ||anchor - negative||
    # We compute squared distances to avoid sqrt (which is expensive)
    diff_pos = anchor - positive
    diff_neg = anchor - negative
    dist_pos_sq = tl.sum(diff_pos * diff_pos, axis=-1)
    dist_neg_sq = tl.sum(diff_neg * diff_neg, axis=-1)

    # Compute triplet loss: max(0, margin - (dist_pos - dist_neg))
    # We compute the margin term directly
    margin_term = margin - (dist_pos_sq - dist_neg_sq)
    loss = tl.where(margin_term > 0, margin_term, 0.0)

    # Reduce the loss over the batch (sum across batch dimension)
    # We sum the loss for each element and then accumulate
    # Since we are doing per-sample loss, we accumulate in a shared variable
    # But we cannot reduce across the batch in a single kernel without reduction
    # Instead, we return per-sample loss and let the outer loop reduce

    # Store per-sample loss
    tl.store(loss + offsets, loss, mask=mask)


@triton.jit
def reduce_loss_kernel(
    loss_ptr,        # pointer to per-sample loss
    batch_size,      # total batch size
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    # Load per-sample losses
    loss = tl.load(loss_ptr + offsets, mask=mask, other=0.0)

    # Sum over batch dimension
    total_loss = tl.sum(loss, axis=0)

    # Store the final loss
    tl.store(total_loss, total_loss, mask=mask)


def triton_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0):
    """
    Custom implementation of Triplet Margin Loss using Triton kernels.
    This replaces the PyTorch triplet margin loss with fused, optimized kernels.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "All tensors must be on CUDA."
    assert anchor.shape == positive.shape == negative.shape, "All inputs must have the same shape."
    
    batch_size, dim = anchor.shape

    # Ensure tensors are contiguous
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    # Allocate output tensor (we will compute loss as scalar)
    # We compute per-sample loss and then reduce to scalar
    # Use FP16 for better memory bandwidth and tensor core utilization
    # We use BF16 for better performance on A100 Tensor Cores

    # Convert to BF16 for Tensor Core acceleration
    anchor_bf16 = anchor.to(torch.bfloat16)
    positive_bf16 = positive.to(torch.bfloat16)
    negative_bf16 = negative.to(torch.bfloat16)

    # Compute per-sample triplet loss using Triton kernels
    # First kernel computes pairwise distances and per-sample loss
    # We use BLOCK_SIZE = 256 for good occupancy and memory coalescing

    # Define grid for first kernel
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch first kernel to compute per-sample loss
    # Output is a tensor of shape (batch_size,) storing per-sample loss
    per_sample_loss = torch.empty(batch_size, dtype=torch.bfloat16, device=anchor.device)

    # Launch triplet margin loss kernel
    triplet_margin_loss_kernel[grid](
        anchor_bf16.data_ptr(),
        positive_bf16.data_ptr(),
        negative_bf16.data_ptr(),
        batch_size,
        dim,
        margin,
        BLOCK_SIZE=256,
    )

    # Second kernel reduces the per-sample loss to scalar
    # We use a single block to sum over the batch
    reduce_grid = lambda meta: (1,)
    reduce_loss_kernel[reduce_grid](
        per_sample_loss.data_ptr(),
        batch_size,
        BLOCK_SIZE=128,
    )

    # Convert back to FP32 for final output (loss is scalar)
    final_loss = per_sample_loss.sum().item()
    return final_loss


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Replace PyTorch triplet loss with custom Triton kernel
        return triton_triplet_loss(anchor, positive, negative, margin=self.margin)