import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------- Triton kernel ---------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def triplet_margin_loss_kernel(
    a_ptr, p_ptr, n_ptr, out_ptr, margin, n_elements, BLOCK_SIZE: tl.constexpr
):
    """
    Computes TripletMarginLoss for a single batch element.
    Each program instance handles one sample.
    """
    batch_idx = tl.program_id(0)

    # Accumulators for squared distances
    acc_ap = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_an = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the feature dimension in chunks of BLOCK_SIZE
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + batch_idx * n_elements + offsets, mask=mask, other=0.0)
        p = tl.load(p_ptr + batch_idx * n_elements + offsets, mask=mask, other=0.0)
        n = tl.load(n_ptr + batch_idx * n_elements + offsets, mask=mask, other=0.0)

        diff_ap = a - p
        diff_an = a - n

        acc_ap += diff_ap * diff_ap
        acc_an += diff_an * diff_an

    # Reduce within the block
    sum_ap = tl.sum(acc_ap)
    sum_an = tl.sum(acc_an)

    # Compute loss for this sample
    loss = tl.max(0.0, tl.sqrt(sum_ap) - tl.sqrt(sum_an) + margin)

    # Store result
    tl.store(out_ptr + batch_idx, loss)


def triplet_margin_loss_torch(anchor, positive, negative, margin):
    """
    Wrapper that calls the Triton kernel and returns the mean loss
    across the batch.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    batch_size = anchor.shape[0]
    n_elements = anchor.shape[1]

    out = torch.empty(batch_size, device=anchor.device, dtype=torch.float32)

    grid = lambda meta: (batch_size,)

    triplet_margin_loss_kernel[grid](
        anchor, positive, negative, out, margin, n_elements
    )

    return out.mean()


# --------------------------- Model ----------------------------------------

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks
    using a custom Triton kernel for improved performance.
    """

    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triplet_margin_loss_torch(anchor, positive, negative, self.margin)