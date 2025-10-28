An expert in writing optimized Triton kernels for GPU computing.
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Pointers to additional tensors for fusion
    linear_bias_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    custom_bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Fusion parameters
    bn_eps,
    divide_value,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused Triton kernel for MatMul, BatchNorm, Bias Addition, Division, and Swish activation.
    Computes: C = swish((batch_norm(A @ B.T + linear_bias) + custom_bias) / divide_value)
    This kernel is optimized for the inference path of BatchNorm.
    """
    # Program IDs
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = (pid // num_pid_n) % num_pid_m
    pid_n = pid % num_pid_n

    # Rematerialize pids for group-level calculations
    pid_m_group = pid_m // GROUP_SIZE_M
    pid_n_group = pid_n

    # Create ranges for the first block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers to A and B matrices
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        
        # Perform matrix multiplication for the current block and accumulate
        accumulator += tl.dot(a, tl.trans(b))
        
        # Advance pointers for the next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # --- Fused Operations Start Here ---
    
    # 1. Add Linear Bias
    linear_bias = tl.load(linear_bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    accumulator += linear_bias[None, :]

    # 2. Batch Normalization (Inference)
    bn_mean = tl.load(bn_mean_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    bn_var = tl.load(bn_var_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    bn_weight = tl.load(bn_weight_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    bn_bias = tl.load(bn_bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)

    # Normalize
    inv_stddev = 1.0 / tl.sqrt(bn_var + bn_eps)
    normalized = (accumulator - bn_mean[None, :]) * inv_stddev[None, :]
    # Scale and shift
    bn_out = normalized * bn_weight[None, :] + bn_bias[None, :]

    # 3. Add Custom Bias
    # Custom bias is a single element tensor, so we load it once.
    custom_bias = tl.load(custom_bias_ptr)
    bias_added = bn_out + custom_bias
    
    # 4. Divide
    divided = bias_added / divide_value
    
    # 5. Swish Activation
    # swish(x) = x * sigmoid(x)
    result = divided * tl.sigmoid(divided)

    # --- Fused Operations End ---

    # Write the final result to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


def triton_fused_op(
    x, matmul_weight, matmul_bias,
    bn, custom_bias, divide_value
):
    """
    Wrapper function for the fused Triton kernel.
    Handles tensor preparations, grid definition, and kernel launch.
    """
    # Ensure inputs are on CUDA and contiguous
    assert all(t.is_cuda for t in [x, matmul_weight, matmul_bias, bn.running_mean, bn.running_var, bn.weight, bn.bias, custom_bias])
    
    x = x.contiguous()
    matmul_weight = matmul_weight.contiguous()
    
    # Extract dimensions
    M, K = x.shape
    N, _ = matmul_weight.shape

    # Allocate output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Grid definition
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch the kernel
    fused_kernel[grid](
        # Pointers
        x, matmul_weight, c,
        matmul_bias,
        bn.running_mean, bn.running_var, bn.weight, bn.bias,
        custom_bias,
        # Dimensions
        M, N, K,
        # Strides
        x.stride(0), x.stride(1),
        matmul_weight.stride(0), matmul_weight.stride(1),
        c.stride(0), c.stride(1),
        # Fusion parameters
        bn.eps,
        divide_value,
    )
    return c


class ModelNew(nn.Module):
    """
    Optimized model using a single fused Triton kernel for MatMul, BatchNorm,
    Bias Addition, Division, and Swish activation.
    
    NOTE: This implementation is optimized for inference (`.eval()` mode) as
    the Triton kernel implements the inference path of BatchNorm and does not
    update running statistics.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        if self.training:
            # Fallback to original PyTorch implementation during training
            # because the fused kernel is for inference only.
            y = self.matmul(x)
            y = self.bn(y)
            y = y + self.bias
            y = y / self.divide_value
            y = y * torch.sigmoid(y)
            return y
        
        # Use the optimized Triton kernel for inference
        return triton_fused_op(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.bn,
            self.bias,
            self.divide_value
        )