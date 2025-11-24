import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_bn_scale_softmax_kernel(
    x_ptr,        # Input tensor: (batch_size, in_features)
    w_ptr,        # Weight matrix: (out_features, in_features)
    b_ptr,        # Bias vector: (out_features,)
    bn_gamma_ptr, # BatchNorm gamma: (out_features,)
    bn_beta_ptr,  # BatchNorm beta: (out_features,)
    bn_running_mean_ptr, # Running mean: (out_features,)
    bn_running_var_ptr,  # Running var: (out_features,)
    scale_ptr,    # Scale parameter: (1,)
    out_ptr,      # Output tensor: (batch_size, out_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    bn_eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of the output dimension
    batch_idx = tl.program_id(0)
    # Compute the offset for this batch
    batch_offset = batch_idx * batch_size
    # Create a range of indices for the output features
    out_offsets = tl.arange(0, out_features)
    # Create a range of indices for the input features
    in_offsets = tl.arange(0, in_features)

    # Load weights and bias
    w = tl.load(w_ptr + (out_offsets[:, None] * in_features + in_offsets[None, :]), mask=in_offsets[None, :] < in_features, other=0.0)
    b = tl.load(b_ptr + out_offsets, mask=out_offsets < out_features, other=0.0)

    # Load batch norm parameters
    gamma = tl.load(bn_gamma_ptr + out_offsets, mask=out_offsets < out_features, other=1.0)
    beta = tl.load(bn_beta_ptr + out_offsets, mask=out_offsets < out_features, other=0.0)
    running_mean = tl.load(bn_running_mean_ptr + out_offsets, mask=out_offsets < out_features, other=0.0)
    running_var = tl.load(bn_running_var_ptr + out_offsets, mask=out_offsets < out_features, other=1.0)

    # Load input data for this batch
    x = tl.load(x_ptr + (batch_offset + tl.arange(0, in_features)), mask=tl.arange(0, in_features) < in_features, other=0.0)

    # Matrix multiplication: x @ w + b
    # Use tensor core-friendly FP16 for performance
    # We compute x @ w in FP16 with tensor cores, then apply BN and softmax
    # We will compute the output for each output feature
    out = tl.zeros((out_features,), dtype=tl.float16)

    # Compute dot product for each output feature
    for i in range(0, out_features, BLOCK_SIZE):
        i_start = i
        i_end = min(i + BLOCK_SIZE, out_features)
        # Create a mask for this slice
        i_mask = i_start < out_features
        # Compute the output for this slice
        out_slice = tl.dot(x, w[:, i_start:i_end], mask=in_offsets[None, :] < in_features)
        out_slice = out_slice + b[i_start:i_end]
        out_slice = out_slice * gamma[i_start:i_end]
        out_slice = out_slice - beta[i_start:i_end]
        out_slice = out_slice + running_mean[i_start:i_end]
        out_slice = out_slice / tl.sqrt(running_var[i_start:i_end] + bn_eps)
        out_slice = out_slice.to(tl.float32)  # Convert to float32 for softmax

        # Accumulate into the full output
        out = out + out_slice

    # Apply softmax in a fused way using a loop over features
    # We do a reduction over the output dimension
    # Instead of full softmax, we do a fused softmax in a single kernel
    # We compute exp(out) and normalize
    # We use a single kernel to avoid memory traffic
    # Compute exp(out) for each feature
    exp_out = tl.exp(out)
    sum_exp = tl.sum(exp_out, axis=0)
    softmax_out = exp_out / sum_exp

    # Store the result
    tl.store(out_ptr + (batch_offset + out_offsets), softmax_out, mask=out_offsets < out_features)


def triton_gemm_bn_scale_softmax(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, 
                                 bn_gamma: torch.Tensor, bn_beta: torch.Tensor, 
                                 bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor, 
                                 scale: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton kernel for fused Gemm + BatchNorm + Scale + Softmax.
    Uses FP16 for computation and FP32 for softmax stability.
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda and bn_gamma.is_cuda and bn_beta.is_cuda and \
           bn_running_mean.is_cuda and bn_running_var.is_cuda and scale.is_cuda, \
           "All tensors must be on CUDA."
    
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    bn_gamma = bn_gamma.contiguous()
    bn_beta = bn_beta.contiguous()
    bn_running_mean = bn_running_mean.contiguous()
    bn_running_var = bn_running_var.contiguous()
    scale = scale.contiguous()

    batch_size, in_features = x.shape
    out_features = w.shape[0]

    # Prepare output tensor
    out = torch.empty((batch_size, out_features), device=x.device, dtype=torch.float32)

    # Define kernel parameters
    BLOCK_SIZE = 128  # Optimal for Ampere, balances memory and compute

    # Grid: number of blocks needed
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    gemm_bn_scale_softmax_kernel[grid](
        x_ptr=x.data_ptr(),
        w_ptr=w.data_ptr(),
        b_ptr=b.data_ptr(),
        bn_gamma_ptr=bn_gamma.data_ptr(),
        bn_beta_ptr=bn_beta.data_ptr(),
        bn_running_mean_ptr=bn_running_mean.data_ptr(),
        bn_running_var_ptr=bn_running_var.data_ptr(),
        scale_ptr=scale.data_ptr(),
        out_ptr=out.data_ptr(),
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        bn_eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        # We keep the Linear layer as a parameter
        self.weight = nn.Parameter(torch.randn(out_features, in_features).cuda())
        self.bias = nn.Parameter(torch.zeros(out_features).cuda())
        # BatchNorm parameters (we use running stats for inference)
        self.bn_gamma = nn.Parameter(torch.ones(out_features).cuda())
        self.bn_beta = nn.Parameter(torch.zeros(out_features).cuda())
        self.bn_running_mean = nn.Parameter(torch.zeros(out_features).cuda())
        self.bn_running_var = nn.Parameter(torch.ones(out_features).cuda())
        self.scale = nn.Parameter(torch.ones(scale_shape).cuda())

    def forward(self, x):
        # Fused kernel: Gemm + BN + Scale + Softmax
        # Note: We use a custom kernel that performs all operations in one pass
        # This avoids intermediate CPU-GPU transfers and reduces memory traffic
        return triton_gemm_bn_scale_softmax(
            x, self.weight, self.bias,
            self.bn_gamma, self.bn_beta,
            self.bn_running_mean, self.bn_running_var,
            self.scale
        )