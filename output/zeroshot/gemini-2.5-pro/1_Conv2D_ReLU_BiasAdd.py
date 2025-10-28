import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        # A100 specific configs with larger tile sizes
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W_in', 'K', 'R', 'S', 'stride_h', 'stride_w', 'pad_h', 'pad_w'],
)
@triton.jit
def conv_relu_bias_kernel(
    # Pointers to Tensors
    X, W, B, Y,
    # Dimensions
    N, C, H, W_in, K, R, S,
    stride_h, stride_w, pad_h, pad_w, P, Q,
    # Strides for accessing tensors
    X_stride_n, X_stride_c, X_stride_h, X_stride_w,
    W_stride_k, W_stride_c, W_stride_r, W_stride_s,
    Y_stride_k, Y_stride_npq,
    # Meta-parameters for tuning
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused kernel for Conv2D + Bias + ReLU.
    This kernel is implemented as a matrix multiplication (GEMM) without
    materializing the im2col matrix.
    - Y (output) is viewed as a 2D matrix of shape (K, N*P*Q).
    - W (weights) is viewed as a 2D matrix of shape (K, C*R*S).
    - X (input) is implicitly gathered to form the second operand of the GEMM.
    Each program instance computes a (BLOCK_M, BLOCK_N) tile of the output Y.
    """
    # Program ID and grid logic
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N * P * Q, BLOCK_N)
    
    # Grouping program IDs for better L2 cache locality
    pid_group = pid // GROUP_SIZE_M
    group_id = pid % GROUP_SIZE_M
    
    pid_m_group = pid_group // num_pid_n
    pid_n = pid_group % num_pid_n
    
    pid_m = pid_m_group * GROUP_SIZE_M + group_id

    # Offsets for the M (output channels) and N (spatial locations) dimensions
    offs_k = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_npq = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Unpack the flat (N, P, Q) offsets into individual dimensions
    offs_n = offs_npq // (P * Q)
    rem_pq = offs_npq % (P * Q)
    offs_p = rem_pq // Q
    offs_q = rem_pq % Q

    # Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the reduction dimension (C*R*S) in blocks of BLOCK_K
    crs_dim = C * R * S
    for k_red_start in range(0, crs_dim, BLOCK_K):
        offs_k_red = k_red_start + tl.arange(0, BLOCK_K)
        
        # Unpack flat reduction index into (c, r, s)
        offs_c = offs_k_red // (R * S)
        rem_rs = offs_k_red % (R * S)
        offs_r = rem_rs // S
        offs_s = rem_rs % S

        # --- Load weight tile (BLOCK_M, BLOCK_K) ---
        w_ptrs = W + (offs_k[:, None] * W_stride_k + offs_c[None, :] * W_stride_c +
                      offs_r[None, :] * W_stride_r + offs_s[None, :] * W_stride_s)
        mask_w = (offs_k[:, None] < K) & (offs_k_red[None, :] < crs_dim)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # --- Load input tile (BLOCK_K, BLOCK_N) ---
        # Calculate input coordinates (h_in, w_in)
        h_in = offs_p[None, :] * stride_h + offs_r[:, None] - pad_h
        w_in = offs_q[None, :] * stride_w + offs_s[:, None] - pad_w
        
        x_ptrs = X + (offs_n[None, :] * X_stride_n + offs_c[:, None