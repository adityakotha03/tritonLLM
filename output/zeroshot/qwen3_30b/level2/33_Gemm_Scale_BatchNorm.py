import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr
):
    # Matrix multiplication: C = A @ B
    # A: [M, K], B: [K, N], C: [M, N]
    # All tensors are 2D with leading dim first
    pid = tl.program_id(0) * GROUP_SIZE_M + tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    if pid_m >= grid_m or pid_n >= grid_n:
        return

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # Create masks to avoid out-of-bounds access
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load A and B
    offs_am = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    offs_bk = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a = tl.load(a_ptr + offs_am, mask=mask_m[:, None], other=0.0)
    b = tl.load(b_ptr + offs_bk, mask=mask_n[None, :], other=0.0)

    # Accumulate result
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load next block
        a_block = tl.load(
            a_ptr + offs_am + k * BLOCK_K * stride_ak,
            mask=mask_m[:, None] & (offs_k[None, :] < (k + 1) * BLOCK_K),
            other=0.0
        )
        b_block = tl.load(
            b_ptr + offs_bk + k * BLOCK_K * stride_bk,
            mask=mask_n[None, :] & (offs_k[:, None] < (k + 1) * BLOCK_K),
            other=0.0
        )
        # Perform matrix multiplication
        accumulator += tl.dot(a_block, b_block, allow_tf32=True)
    
    # Apply scaling
    scale = tl.load(SCALE + offs_n, mask=mask_n, other=0.0)
    accumulator = accumulator * scale[None, :]

    # Store output
    offs_cm = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr + offs_cm, accumulator, mask=mask)


@triton.jit
def batch_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr,
    N, C,
    stride_x, stride_w, stride_b, stride_m, stride_v,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Batch norm: y = (x - mean) / sqrt(var + eps) * weight + bias
    # x: [N, C], weight: [C], bias: [C], mean: [C], var: [C]
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    # Load input and params
    x = tl.load(x_ptr + offsets * stride_x, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets * stride_w, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets * stride_b, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offsets * stride_m, mask=mask, other=0.0)
    var = tl.load(var_ptr + offsets * stride_v, mask=mask, other=0.0)

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)
    # Scale and shift
    y = x_norm * weight + bias

    # Store result
    tl.store(x_ptr + offsets * stride_x, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features).t().contiguous())
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.bn_mean = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bn_var = nn.Parameter(torch.ones(out_features), requires_grad=False)

    def forward(self, x):
        # Ensure contiguous input
        x = x.contiguous()
        # Shape: [batch_size, in_features]
        batch_size, in_features = x.shape
        out_features = self.out_features

        # Prepare input for GEMM
        x_ptr = x.data_ptr()
        w_ptr = self.gemm_weight.data_ptr()
        out_ptr = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype).data_ptr()

        # Strides
        stride_x = x.stride(0)  # batch
        stride_w = self.gemm_weight.stride(0)  # out_features
        stride_out = x.stride(0)  # batch

        # GEMM with fused scaling and BN
        M, N, K = batch_size, out_features, in_features
        # Use 512 for high occupancy on A100
        BLOCK_M, BLOCK_N, BLOCK_K = 512, 512, 128
        GROUP_SIZE_M = 8

        # Grid setup
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
            meta["GROUP_SIZE_M"]
        )

        # Launch fused matmul + scale + BN
        matmul_kernel[
            grid
        ](
            x_ptr, w_ptr, out_ptr,
            M, N, K,
            stride_x, stride_w,
            stride_w, stride_out,
            stride_out, stride_out,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            ACTIVATION="none",
            SCALE=self.scale.data_ptr(),
            EPS=self.eps
        )

        # Now do BN in-place on the output (fused kernel)
        out = torch.from_numpy(
            torch.tensor(out_ptr, dtype=x.dtype, device=x.device).view(batch_size, out_features)
        ).contiguous()
        
        # Apply batch norm in-place
        # Get pointers
        x_ptr = out.data_ptr()
        weight_ptr = self.bn_weight.data_ptr()
        bias_ptr = self.bn_bias.data_ptr()
        mean_ptr = self.bn_mean.data_ptr()
        var_ptr = self.bn_var.data_ptr()

        # Strides
        stride_x = out.stride(0)
        stride_w = self.bn_weight.stride(0)
        stride_b = self.bn_bias.stride(0)
        stride_m = self.bn_mean.stride(0)
        stride_v = self.bn_var.stride(0)

        # Block size
        BLOCK_SIZE = 128  # power of 2, good for register use

        # Grid
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

        # Launch BN kernel
        batch_norm_kernel[
            grid
        ](
            x_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr,
            N, out_features,
            stride_x, stride_w, stride_b, stride_m, stride_v,
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Update running mean/var if in train mode
        with torch.no_grad():
            # Update running stats
            if self.training:
                x_mean = out.mean(dim=0)
                x_var = out.var(dim=0, unbiased=False)
                self.bn_mean = self.momentum * self.bn_mean + (1 - self.momentum) * x_mean
                self.bn_var = self.momentum * self.bn_var + (1 - self.momentum) * x_var

        return out