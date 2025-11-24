import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_sigmoid_kernel(
    x_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    out_ptr,
    n_samples: tl.constexpr,
    n_input: tl.constexpr,
    n_hidden: tl.constexpr,
    n_output: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples

    # Load input x (batch x input_size)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load first linear weights (input to hidden)
    w1 = tl.load(w1_ptr + (offsets[:, None] * n_hidden + tl.arange(0, n_hidden)[None, :]), 
                 mask=offsets[:, None] < n_input, other=0.0)
    # Compute linear1 output: x @ w1 + b1
    xw1 = tl.dot(x, w1)  # (batch x hidden)
    # Add bias
    b1 = tl.load(b1_ptr + offsets, mask=mask, other=0.0)
    xw1 = xw1 + b1[:, None]

    # Apply sigmoid activation
    # We use a fused sigmoid with a stable computation
    exp_xw1 = tl.exp(xw1 - tl.max(xw1, axis=-1, keepdim=True))
    sigmoid_val = exp_xw1 / tl.sum(exp_xw1, axis=-1, keepdim=True)
    # Store intermediate result
    tl.store(sigmoid_val + offsets, sigmoid_val, mask=mask)

    # Now compute second linear layer: sigmoid(x) @ w2 + b2
    # Load second weights (hidden to output)
    w2 = tl.load(w2_ptr + (tl.arange(0, n_output)[None, :] * n_hidden + tl.arange(0, n_hidden)[:, None]), 
                 mask=tl.arange(0, n_hidden)[:, None] < n_hidden, other=0.0)
    # Compute linear2 output
    out = tl.dot(sigmoid_val, w2)  # (batch x output)
    b2 = tl.load(b2_ptr + offsets, mask=mask, other=0.0)
    out = out + b2[:, None]

    # Compute logsumexp over features (dim=1)
    out_max = tl.max(out, axis=1, keepdim=True)
    exp_out = tl.exp(out - out_max)
    logsumexp_val = tl.log(tl.sum(exp_out, axis=1)) + out_max
    # Store final result
    tl.store(out_ptr + offsets, logsumexp_val, mask=mask)


@triton.jit
def matmul_sigmoid_fused_kernel(
    x_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    out_ptr,
    n_samples: tl.constexpr,
    n_input: tl.constexpr,
    n_hidden: tl.constexpr,
    n_output: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_samples

    # Load input x (batch x input_size)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load first linear weights (input to hidden)
    w1 = tl.load(w1_ptr + (offsets[:, None] * n_hidden + tl.arange(0, n_hidden)[None, :]), 
                 mask=offsets[:, None] < n_input, other=0.0)
    # Compute linear1 output: x @ w1 + b1
    xw1 = tl.dot(x, w1)  # (batch x hidden)
    b1 = tl.load(b1_ptr + offsets, mask=mask, other=0.0)
    xw1 = xw1 + b1[:, None]

    # Apply sigmoid activation via stable log-sum-exp trick
    # We avoid direct sigmoid by using log-sum-exp for stability
    # Instead, we compute logsumexp of exp(xw1) which is equivalent to sigmoid
    # But we actually want sigmoid(xw1) = 1 / (1 + exp(-xw1))
    # So we use: log(1 + exp(xw1)) - xw1
    # But this is not exactly sigmoid. Instead, we compute sigmoid directly with stable exponentiation
    # We use a stable sigmoid: sigmoid = 1 / (1 + exp(-xw1))
    # We compute exp(-xw1) safely
    neg_xw1 = -xw1
    exp_neg_xw1 = tl.exp(neg_xw1 - tl.max(neg_xw1, axis=-1, keepdim=True))
    sigmoid_val = 1.0 / (1.0 + exp_neg_xw1)
    # Store intermediate result
    tl.store(sigmoid_val + offsets, sigmoid_val, mask=mask)

    # Now compute second linear layer: sigmoid(x) @ w2 + b2
    w2 = tl.load(w2_ptr + (tl.arange(0, n_output)[None, :] * n_hidden + tl.arange(0, n_hidden)[:, None]), 
                 mask=tl.arange(0, n_hidden)[:, None] < n_hidden, other=0.0)
    out = tl.dot(sigmoid_val, w2)  # (batch x output)
    b2 = tl.load(b2_ptr + offsets, mask=mask, other=0.0)
    out = out + b2[:, None]

    # Compute logsumexp over features (dim=1)
    out_max = tl.max(out, axis=1, keepdim=True)
    exp_out = tl.exp(out - out_max)
    logsumexp_val = tl.log(tl.sum(exp_out, axis=1)) + out_max
    tl.store(out_ptr + offsets, logsumexp_val, mask=mask)


def triton_model_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
):
    """
    Forward pass using custom Triton kernels.
    """
    assert x.is_cuda and w1.is_cuda and b1.is_cuda and w2.is_cuda and b2.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    w1 = w1.contiguous()
    b1 = b1.contiguous()
    w2 = w2.contiguous()
    b2 = b2.contiguous()

    n_samples = x.shape[0]
    n_input = x.shape[1]
    n_hidden = w1.shape[1]
    n_output = w2.shape[0]

    # Output tensor of shape (batch_size,)
    out = torch.empty(n_samples, device=x.device, dtype=torch.float32)

    # Define block size for optimal performance on A100
    BLOCK_SIZE = 128

    # Grid size: number of blocks needed
    grid = lambda meta: ((n_samples + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    matmul_sigmoid_fused_kernel[grid](x, w1, b1, w2, b2, out, 
                                      n_samples=n_samples, 
                                      n_input=n_input, 
                                      n_hidden=n_hidden, 
                                      n_output=n_output, 
                                      BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.w1 = torch.randn(hidden_size, input_size, device='cuda', dtype=torch.float16)
        self.b1 = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
        self.w2 = torch.randn(output_size, hidden_size, device='cuda', dtype=torch.float16)
        self.b2 = torch.randn(output_size, device='cuda', dtype=torch.float16)

    def forward(self, x):
        # Use custom Triton kernel for fused matmul + sigmoid + matmul + logsumexp
        return triton_model_forward(x, self.w1, self.b1, self.w2, self.b2)