import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    normalized_size: tl.constexpr,
    stride, stride_out,
    num_tokens: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * stride
    out_offset = pid * stride_out

    # Load input block
    x_offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < normalized_size
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / normalized_size

    # Compute variance (with mean subtraction)
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / normalized_size
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine transform
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    b = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    normed = (x - mean) * inv_std
    output = normed * w + b

    # Store result
    tl.store(out_ptr + out_offset + tl.arange(0, BLOCK_SIZE), output, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on GPU."

        # Reshape input into (*batch, feature_dim) where feature_dim is the normalized part
        feature_dim = self.normalized_shape
        input_shape = x.shape
        batch_shape = input_shape[:-len(feature_dim)]
        x_reshaped = x.view(-1, *feature_dim)
        batch_size = x_reshaped.size(0)

        # Flatten the normalized dimensions for kernel processing
        x_flat = x_reshaped.view(batch_size, -1)  # (B, D), D = prod(normalized_shape)
        D = x_flat.size(1)

        # Ensure D is power of 2 or set appropriate block size
        BLOCK_SIZE = triton.next_power_of_2(D)

        # Output buffer
        out = torch.empty_like(x_flat)

        # Launch kernel
        def grid(meta): return (batch_size,)

        layer_norm_kernel[grid](
            x_flat, self.weight.data, self.bias.data, out,
            normalized_size=D,
            stride=D,
            stride_out=D,
            num_tokens=batch_size,
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Reshape back to original
        out = out.view(input_shape)
        return out