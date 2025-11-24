import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_sub_gelu_logsumexp_avg_residual_kernel(
    x_ptr, weight_ptr, bias_ptr, subtract_param_ptr, residual_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_GELU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    weight_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(weight_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        weight_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    # Subtract parameter (per output feature)
    subtract_val = tl.load(subtract_param_ptr + offs_n, mask=offs_n < N, other=0.0)
    accumulator -= subtract_val[None, :]

    # GlobalAvgPool: average over sequence dim (here M is treated as seq len)
    # But note: we are doing this after matmul, so we need to reduce over M
    # However, we are in a tile-by-tile computation. So we must defer reduction.
    # Instead, we will do the reduction in a separate fused step later.
    # But since we are fusing everything, let's restructure.

    # Actually, let's refactor: we'll do the full matmul, then in a separate kernel do the reductions and activations.
    # But we can fuse the pointwise ops after reduction.

    # Since the reduction dims are not trivial to fuse into matmul without tiling issues,
    # we instead create a separate kernel for post-processing.

    # So this kernel only does: matmul + bias + subtract
    # Then we'll write a second kernel for: avgpool -> logsumexp -> gelu -> add residual

    # Store intermediate result
    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + offs_out_m[:, None] * stride_om + offs_out_n[None, :] * stride_on
    tl.store(out_ptrs, accumulator, mask=(offs_out_m[:, None] < M) & (offs_out_n[None, :] < N))


@triton.jit
def reduce_and_activate_kernel(
    x_ptr, residual_ptr, out_ptr,
    M, N,
    stride_xn, stride_resm, stride_resn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_GELU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr
):
    pid_n = tl.program_id(0)

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Load the entire column across all M rows for each n
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # GlobalAvgPool: mean over M dimension
    sum_val = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask = mask_n[None, :] & (offs_m[:, None] < M)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xn + offs_n[None, :], mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)

    mean_val = sum_val / M

    # LogSumExp: log(sum(exp(x))) but over M? Wait, original is: torch.logsumexp(x, dim=1, keepdim=True)
    # But after avgpool, x has shape (M, 1)? Actually no: after mean(x, dim=1, keepdim=True), it's (M, 1)
    # But in our case, we have (M, N), and we are reducing over dim=1 (M), so output is (1, N) per row?
    # Actually, no: the input to logsumexp is (M, N), and we reduce over dim=1 -> (M, 1)? No: dim=1 is N!
    # Let's recheck: x = torch.mean(x, dim=1, keepdim=True) -> dim=1 is the second dim, which is N (features).
    # But that doesn't make sense: usually we average over sequence length (dim=1 if input is (B, S, D)).
    # In this model, input x is (batch_size, in_features) = (2048, 8192)
    # After gemm: (2048, 8192) @ (8192, 8192) -> (2048, 8192)
    # Then subtract: (2048, 8192)
    # Then torch.mean(x, dim=1, keepdim=True): mean over dim=1 (8192) -> (2048, 1)
    # Then logsumexp over dim=1: but now dim=1 is 1, so logsumexp((2048,1), dim=1) -> (2048,1)
    # But that's just the value itself? Because log(sum(exp(x))) of one element is x.

    # Actually: logsumexp over dim=1 of a (2048,1) tensor gives (2048,1), yes, and it's equal to the input.
    # But that seems redundant. However, we'll follow the code.

    # But wait: the mean reduces to (2048,1), then logsumexp(dim=1) on (2048,1) -> (2048,1), yes.

    # So after mean: x becomes (M, 1), but in our kernel we still have (M, N). So we must have already reduced N to 1?

    # Actually, the GlobalAvgPool reduces dim=1 (N) to 1, so output is (M, 1). Then logsumexp over dim=1 again? That would reduce the 1 to scalar? No, keepdim=True.

    # But the code says: torch.mean(x, dim=1, keepdim=True) -> (M, 1)
    # Then torch.logsumexp(x, dim=1, keepdim=True) -> (M, 1)

    # So we need to reduce N to 1 first.

    # Therefore, we should first reduce over N dimension to get (M, 1), then logsumexp over dim=1 (which is now the only dim besides M) -> (M,1)

    # But logsumexp over dim=1 of a (M,1) tensor is just the same as the input? Because log(sum(exp(x_i))) for one element is x_i.

    # So logsumexp is redundant here. But we'll keep it.

    # Actually, let's re-read: the model does:
    #   x = torch.mean(x, dim=1, keepdim=True)   # (M, N) -> (M, 1)
    #   x = torch.logsumexp(x, dim=1, keepdim=True)  # (M, 1) -> (M, 1) -> same as input

    # So we can skip logsumexp.

    # But to be safe, we'll implement it.

    # However, our kernel is designed for (M, N) input. We need to reduce N to 1.

    # So we must do: first reduce over N to get (M, 1), then logsumexp over dim=1.

    # But we are only reducing over M in the above code? That's wrong.

    # Let's correct: the GlobalAvgPool is over dim=1, which is N (the feature dim), so we reduce N to 1.

    # So we need to compute: mean over N (dim=1) for each row -> (M, 1)

    # But in our kernel, we are processing by tiles in N. So we need to reduce over N.

    # We'll change strategy: we write a kernel that reduces over N (dim=1) to 1.

    # But this kernel is not scalable. Instead, we write a separate fused kernel for the entire post-processing.

    # Given complexity, we split into two kernels:
    # 1. matmul + bias + subtract -> output (M, N)
    # 2. a fused kernel that does: mean(dim=1) -> logsumexp(dim=1) -> gelu -> add residual

    # But residual is (M, N) and after reduction we have (M, 1), so we cannot add.

    # Wait: residual is the original input, which is (M, K) = (2048, 8192), but output of gemm is (2048, 8192), then reduced to (2048,1), then we add residual (2048,8192)? That would broadcast.

    # But the code: 
    #   original_x = x.clone().detach()   # (2048, 8192)
    #   ... 
    #   x = torch.nn.functional.gelu(x)   # (2048, 1)
    #   x = x + original_x                 # (2048, 1) + (2048, 8192) -> broadcasts to (2048, 8192)

    # So the final output is (2048, 8192)

    # Therefore, we need to broadcast the (M,1) to (M, N) and then add.

    # So the post-processing kernel should:
    #   - reduce the input (M, N) over dim=1 to (M, 1) [mean]
    #   - apply logsumexp over dim=1 -> (M,1) [but this is redundant]
    #   - apply gelu -> (M,1)
    #   - broadcast to (M, N) and add residual (M, N)

    # We can fuse all that.

    pass  # We abandon this kernel for now and do a simpler fusion.


# Instead, we do a more practical approach: fuse the pointwise operations after the matmul into one kernel, but not with matmul.

# We'll write a kernel for the reduction and activation chain.

@triton.jit
def reduce_and_activate_kernel(
    x_ptr, residual_ptr, out_ptr,
    M, N,
    stride_xn, stride_resm, stride_resn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m
    if offs_m >= M:
        return

    # Load one row of x: (N,)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    x_row = tl.load(x_ptr + offs_m * stride_xn + offs_n, mask=mask_n, other=0.0)

    # GlobalAvgPool: mean over N
    mean_val = tl.sum(x_row) / N

    # LogSumExp over dim=1: but now we have a scalar per row? 
    # But the input to logsumexp is (M,1) after mean, but we are doing it per row.
    # Actually, after mean, we have a scalar per row. Then logsumexp over dim=1 of a scalar is the scalar.
    # So we skip logsumexp.

    # GELU
    x_gelu = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val / 1.41421356))

    # Now broadcast to all N elements and add residual
    residual_row = tl.load(residual_ptr + offs_m * stride_resm + offs_n, mask=mask_n, other=0.0)
    output_row = x_gelu + residual_row

    tl.store(out_ptr + offs_m * stride_resm + offs_n, output_row, mask=mask_n)


def triton_matmul_sub(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, subtract_param: torch.Tensor):
    M, K = x.shape
    K, N = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # 1D launch kernel where each block gets a row of the input
    grid = lambda META: (M,)

    matmul_kernel_sub[grid](
        x, weight, bias, subtract_param, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64,
        BLOCK_N=256,
        BLOCK_K=32,
        GROUP_SIZE=8,
        HAS_BIAS=bias is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_sub(
    x_ptr, weight_ptr, bias_ptr, subtract_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(weight_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        weight_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator = accumulator + bias[None, :]

    # Subtract the parameter
    subtract_val = tl.load(subtract_ptr + offs_n, mask=offs_n < N, other=0.0)
    accumulator = accumulator - subtract_val[None, :]

    offs_out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = output_ptr + offs_out_m[:, None] * stride_om + offs_out_n[None, :] * stride_on
    tl.store(out_ptrs, accumulator, mask=(offs_out_m[:, None] < M) & (offs_out_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 2048}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 4096}, num_stages=1, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def reduce_gelu_residual_kernel(
    x_ptr, residual_ptr, output_ptr,
    M, N,
    stride_xn, stride_resm, stride_resn,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Load the entire row of x (after matmul) of size N
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    x_row = tl.load(x_ptr + pid_m * stride_xn + offs_n, mask=mask_n, other=0.0)

    # GlobalAvgPool: mean over N
    mean_val = tl.sum(x_row) / N

    # LogSumExp over dim=1: not needed as it's a scalar per row, but we do it for correctness
    # But logsumexp of a scalar is the scalar, so skip.

    # GELU
    x_gelu = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val / 1.41421356))

    # Load residual row and add the broadcasted value
    residual_row = tl.load(residual_ptr + pid_m * stride_resm + offs_n, mask=mask_n, other=0.0)
    output_row = x_gelu + residual_row

    tl.store(output_ptr + pid_m * stride_resm + offs_n, output_row, mask=mask_n)


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for fused operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.linear_bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear_weight, a=5**0.5)
        if self.linear_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear_weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.linear_bias, -bound, bound)

    def forward(self, x):
        original_x = x.detach().clone()

        # Triton fused: matmul + bias + subtract
        x = triton_matmul_sub(x, self.linear_weight, self.linear_bias, self.subtract)

        # Triton fused: avgpool (over dim=1) -> gelu -> add residual
        out = torch.empty_like(original_x)
        grid = lambda META: (x.shape[0],)
        reduce_gelu_residual_kernel[grid](
            x, original_x, out,
            x.shape[0], x.shape[1],
            x.stride(0), original_x.stride(0), original_x.stride(1),
            BLOCK_SIZE_N=triton.next_power_of_2(x.shape[1])
        )
        return out