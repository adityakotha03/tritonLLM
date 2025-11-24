import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def squared_l2_distance_kernel(
    anchor_ptr, positive_ptr, negative_ptr,
    output_ptr,
    n_elements, batch_size,
    stride_a_batch, stride_p_batch, stride_n_batch,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute batch index
    batch_idx = tl.program_id(0)
    
    # Pointers to the start of each sample
    a_offset = batch_idx * stride_a_batch
    p_offset = batch_idx * stride_p_batch
    n_offset = batch_idx * stride_n_batch

    # Initialize distance accumulators
    d_pos = 0.0
    d_neg = 0.0

    # Loop over input dimension in blocks
    for start_n in range(0, n_elements, BLOCK_SIZE_N):
        # Compute offsets
        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets_n < n_elements

        # Load chunks
        a = tl.load(anchor_ptr + a_offset + offsets_n, mask=mask, other=0.0)
        p = tl.load(positive_ptr + p_offset + offsets_n, mask=mask, other=0.0)
        n = tl.load(negative_ptr + n_offset + offsets_n, mask=mask, other=0.0)

        # Compute squared differences
        diff_pos = a - p
        diff_neg = a - n
        d_pos += tl.sum(diff_pos * diff_pos)
        d_neg += tl.sum(diff_neg * diff_neg)

    # Compute loss: max(d(a,n) - d(a,p) + margin, 0)
    margin = 1.0
    loss = tl.maximum(0.0, d_neg - d_pos + margin)

    # Store result
    tl.store(output_ptr + batch_idx, loss)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
        anchor = anchor.contiguous()
        positive = positive.contiguous()
        negative = negative.contiguous()

        batch_size = anchor.shape[0]
        n_elements = anchor.numel() // batch_size

        out = torch.empty((batch_size,), dtype=torch.float32, device=anchor.device)

        # Heuristics for block size
        BLOCK_SIZE_N = 1024
        while BLOCK_SIZE_N > n_elements and BLOCK_SIZE_N > 32:
            BLOCK_SIZE_N //= 2

        BLOCK_SIZE = 256
        grid = lambda meta: (batch_size,)

        squared_l2_distance_kernel[grid](
            anchor,
            positive,
            negative,
            out,
            n_elements,
            batch_size,
            anchor.stride(0),
            positive.stride(0),
            negative.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        # Return mean of the loss
        return out.mean()