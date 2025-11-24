import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    anchor_ptr,  # Pointer to anchor tensor
    positive_ptr,  # Pointer to positive tensor
    negative_ptr,  # Pointer to negative tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in each tensor
    margin,  # Margin parameter
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load anchor, positive, and negative values
    a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
    n = tl.load(negative_ptr + offsets, mask=mask, other=0.0)

    # Compute pairwise distances
    ap_dist = a - p
    an_dist = a - n

    # Apply margin
    loss = tl.where(ap_dist - an_dist > margin, ap_dist - an_dist, 0.0)

    # Sum the loss for this block
    block_sum = tl.sum(loss, axis=0)
    tl.store(output_ptr + pid, block_sum, mask=pid < n_elements // BLOCK_SIZE)


def triton_triplet_loss(anchor, positive, negative, margin):
    """
    Custom Triton kernel for Triplet Margin Loss.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    # Compute the number of elements per tensor
    n_elements = anchor.numel()
    # Output is a scalar, so we can use a single element tensor
    output = torch.tensor(0.0).cuda()

    # Determine the number of blocks needed
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = (num_blocks,)
    triplet_loss_kernel[grid](
        anchor, positive, negative, output, n_elements, margin, BLOCK_SIZE=BLOCK_SIZE
    )
    return output.item()


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks using a custom Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin)