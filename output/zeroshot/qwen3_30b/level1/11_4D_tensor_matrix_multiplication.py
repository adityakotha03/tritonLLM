import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_4d_kernel(
    A_ptr, B_ptr, C_ptr,
    b, i, j, l, k,
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_cb, stride_ci, stride_cj, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Assume inputs are in bf16, use tensor cores for maximum performance
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(i, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(k, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(l, BLOCK_SIZE_K)
    num_pid = num_pid_m * num_pid_n * num_pid_k

    # Calculate the pid for M, N, K
    pid_m = pid // (num_pid_n * num_pid_k)
    pid_n = (pid // num_pid_k) % num_pid_n
    pid_k = pid % num_pid_k

    # Offset for the block in M dimension
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Create offsets for M, N, K
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator for the block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, tl.cdiv(l, BLOCK_SIZE_K)):
        # Compute current K block
        current_k = k * BLOCK_SIZE_K
        offs_k = current_k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < l

        # Load A: [BLOCK_SIZE_M, BLOCK_SIZE_K] (b, i, j, l) -> [b, i, j, l]
        offs_m_idx = offs_m[:, None]
        offs_k_idx = offs_k[None, :]
        offs = (offs_m_idx * stride_ai + offs_k_idx * stride_al)
        a = tl.load(A_ptr + offs, mask=(offs_m[:, None] < i) & (offs_k[None, :] < l), other=0.0)

        # Load B: [BLOCK_SIZE_K, BLOCK_SIZE_N] (l, k) -> [l, k]
        offs_k_idx = offs_k[:, None]
        offs_n_idx = offs_n[None, :]
        offs = (offs_k_idx * stride_bl + offs_n_idx * stride_bk)
        b = tl.load(B_ptr + offs, mask=(offs_k[:, None] < l) & (offs_n[None, :] < k), other=0.0)

        # Convert to float32 for accumulation (tensor cores use FP32 accumulator)
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Perform matmul with tensor cores (implicit in Triton's f16/bf16 matmul)
        accumulator += tl.dot(a, b)

    # Apply activation if needed (e.g., relu, but we leave it as identity here)
    if ACTIVATION == "relu":
        accumulator = tl.max(accumulator, 0)

    # Convert back to bf16
    accumulator = accumulator.to(tl.bfloat16)

    # Store output
    offs_m_idx = offs_m[:, None]
    offs_n_idx = offs_n[None, :]
    offs = (offs_m_idx * stride_ci + offs_n_idx * stride_cj)
    mask = (offs_m[:, None] < i) & (offs_n[None, :] < k)
    tl.store(C_ptr + offs, accumulator, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        # Ensure inputs are in bf16 and contiguous
        A = A.to(torch.bfloat16).contiguous()
        B = B.to(torch.bfloat16).contiguous()

        # Output shape
        b, i, j, l = A.shape
        k = B.shape[1]

        # Output tensor
        C = torch.empty(b, i, j, k, dtype=torch.bfloat16, device=A.device)

        # Strides
        stride_ab = A.stride(0)
        stride_ai = A.stride(1)
        stride_aj = A.stride(2)
        stride_al = A.stride(3)
        stride_bl = B.stride(0)
        stride_bk = B.stride(1)
        stride_cb = C.stride(0)
        stride_ci = C.stride(1)
        stride_cj = C.stride(2)
        stride_ck = C.stride(3)

        # Define block sizes tuned for A100 (Tensor Core optimized)
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 64
        GROUP_SIZE_M = 8

        # Number of program instances
        num_pid_m = triton.cdiv(i, BLOCK_SIZE_M)
        num_pid_n = triton.cdiv(k, BLOCK_SIZE_N)
        num_pid_k = triton.cdiv(l, BLOCK_SIZE_K)
        num_pid = num_pid_m * num_pid_n * num_pid_k

        # Grid configuration
        grid = lambda meta: (num_pid,)

        # Launch kernel with autotuning
        matmul_4d_kernel[grid](
            A, B, C,
            b, i, j, l, k,
            stride_ab, stride_ai, stride_aj, stride_al,
            stride_bl, stride_bk,
            stride_cb, stride_ci, stride_cj, stride_ck,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            ACTIVATION="none"
        )

        return C