@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE': 1}, num_stages=2, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, 
                 M, N, K, 
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE: tl.constexpr):
    # Each program computes a BLOCK_M x BLOCK_N tile of C
    pid = tl.program_id(0)
    pid_x = tl.program_id(1)
    # Compute start positions
    row = pid_x * BLOCK_M
    col = pid * BLOCK_N

    # allocate registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # load tile of A
        A = tl.load(A_ptr + (row + tl.arange(0, BLOCK_M))[:, None] * stride_am + (k + tl.arange(0, BLOCK_K)) * stride_ak, 
                    mask=(row + tl.arange(0, BLOCK_M))[:, None] < M & (k + tl.arange(0, BLOCK_K)) < K,
                    other=0.0)
        # load tile of B
        B = tl.load(B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + (col + tl.arange(0, BLOCK_N)) * stride_bn,
                    mask=(k + tl.arange(0, BLOCK_K))[:, None] < K & (col + tl.arange(0, BLOCK_N)) < N,
                    other=0.0)
        # matmul
        acc += tl.dot(A, B)
    # store
    tl.store(C_ptr + (row + tl.arange(0, BLOCK_M))[:, None] * stride_cm + (col + tl.arange(0, BLOCK_N)) * stride_cn,
             acc,
             mask=(row + tl.arange(0, BLOCK_M))[:, None] < M & (col + tl.arange(0, BLOCK_N)) < N)