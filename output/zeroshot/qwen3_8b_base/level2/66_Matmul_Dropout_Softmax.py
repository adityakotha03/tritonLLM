import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_softmax_dropout_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    bias_ptr,  # Pointer to bias tensor
    dropout_mask_ptr,  # Pointer to dropout mask
    n_elements,  # Total number of elements in input/output
    batch_size,  # Batch size
    in_features,  # Input features
    out_features,  # Output features
    dropout_p,  # Dropout probability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    # Compute matmul
    acc = tl.dot(x, w)
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        acc += bias

    # Apply softmax
    max_val = tl.max(acc, axis=1)
    exp_acc = tl.exp(acc - max_val[:, None])
    sum_exp = tl.sum(exp_acc, axis=1)
    softmax = exp_acc / sum_exp[:, None]

    # Apply dropout
    if dropout_mask_ptr is not None:
        dropout_mask = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0.0)
        softmax = softmax * dropout_mask
        softmax = softmax / (1.0 - dropout_p)

    # Store output
    tl.store(out_ptr + offsets, softmax, mask=mask)


def triton_matmul_softmax_dropout(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dropout_mask: torch.Tensor, dropout_p: float):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and dropout_mask.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    dropout_mask = dropout_mask.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_softmax_dropout_kernel[grid](
        x, weight, out, bias, dropout_mask, n_elements,
        x.size(0), x.size(1), weight.size(0), dropout_p,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.dropout_mask = torch.empty((1, in_features), device='cuda', dtype=torch.float32)

    def forward(self, x):
        # Generate dropout mask
        if self.dropout_p > 0:
            self.dropout_mask = torch.bernoulli(torch.full_like(self.dropout_mask, 1 - self.dropout_p))
        else:
            self.dropout_mask = torch.ones_like(self.dropout_mask)

        # Perform matmul, softmax, and dropout in one kernel
        return triton_matmul_softmax_dropout(x, self.weight, self.bias, self.dropout_mask, self.dropout_p)