import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    H, W, C_in, C_out, K, stride,
    batch_size, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr,
    TILE_SIZE_H: tl.constexpr, TILE_SIZE_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c_out = tl.program_id(2)
    pid_c_in = tl.program_id(3)

    # Compute block offsets
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    c_out_offset = pid_c_out * BLOCK_SIZE_C_OUT
    c_in_offset = pid_c_in * BLOCK_SIZE_C_IN

    # Load input tile
    input_ptrs = x_ptr + (tl.arange(0, BLOCK_SIZE_H)[:, None] + h_offset) * W + \
                 (tl.arange(0, BLOCK_SIZE_W)[None, :] + w_offset)
    input_tile = tl.load(input_ptrs, mask=(h_offset + tl.arange(0, BLOCK_SIZE_H)[:, None] < H) &
                                 (w_offset + tl.arange(0, BLOCK_SIZE_W)[None, :] < W), other=0.0)

    # Load weight tile
    weight_ptrs = w_ptr + (tl.arange(0, BLOCK_SIZE_C_OUT)[:, None] + c_out_offset) * (K * K * C_in) + \
                  (tl.arange(0, BLOCK_SIZE_C_IN)[None, :] + c_in_offset) * (K * K) + \
                  (tl.arange(0, K)[:, None] * K + tl.arange(0, K)[None, :]).reshape(1, 1, -1)
    weight_tile = tl.load(weight_ptrs, mask=(c_out_offset + tl.arange(0, BLOCK_SIZE_C_OUT)[:, None] < C_out) &
                                   (c_in_offset + tl.arange(0, BLOCK_SIZE_C_IN)[None, :] < C_in), other=0.0)

    # Compute convolution: (H, W, C_in) x (C_out, C_in, K, K) -> (H, W, C_out)
    conv_result = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C_OUT), dtype=tl.float32)
    for h in range(K):
        for w in range(K):
            input_val = input_tile[:, :, None] * weight_tile[None, None, :, :, h, w]
            conv_result += tl.sum(input_val, axis=2)

    # Store output
    out_ptrs = out_ptr + (tl.arange(0, BLOCK_SIZE_H)[:, None] + h_offset) * W * C_out + \
               (tl.arange(0, BLOCK_SIZE_W)[None, :] + w_offset) * C_out + \
               (tl.arange(0, BLOCK_SIZE_C_OUT)[None, None, :] + c_out_offset)
    tl.store(out_ptrs, conv_result, mask=(h_offset + tl.arange(0, BLOCK_SIZE_H)[:, None] < H) &
                                     (w_offset + tl.arange(0, BLOCK_SIZE_W)[None, :] < W) &
                                     (c_out_offset + tl.arange(0, BLOCK_SIZE_C_OUT)[None, None, :] < C_out))


@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    H, W, C, stride,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    c_offset = pid_c * BLOCK_SIZE_C

    # Load input tile
    input_ptrs = x_ptr + (tl.arange(0, BLOCK_SIZE_H)[:, None] + h_offset) * W * C + \
                 (tl.arange(0, BLOCK_SIZE_W)[None, :] + w_offset) * C + \
                 (tl.arange(0, BLOCK_SIZE_C)[None, None, :] + c_offset)
    input_tile = tl.load(input_ptrs, mask=(h_offset + tl.arange(0, BLOCK_SIZE_H)[:, None] < H) &
                                   (w_offset + tl.arange(0, BLOCK_SIZE_W)[None, :] < W) &
                                   (c_offset + tl.arange(0, BLOCK_SIZE_C)[None, None, :] < C), other=-float('inf'))

    # Compute max pooling
    output = tl.max(input_tile, axis=0)  # Max over spatial dimensions (H, W)

    # Store output
    out_ptrs = out_ptr + (tl.arange(0, BLOCK_SIZE_H)[:, None] + h_offset) * (W // stride) * C + \
               (tl.arange(0, BLOCK_SIZE_W)[None, :] + w_offset) * C + \
               (tl.arange(0, BLOCK_SIZE_C)[None, None, :] + c_offset)
    tl.store(out_ptrs, output, mask=(h_offset + tl.arange(0, BLOCK_SIZE_H)[:, None] < H // stride) &
                                     (w_offset + tl.arange(0, BLOCK_SIZE_W)[None, :] < W // stride) &
                                     (c_offset + tl.arange(0, BLOCK_SIZE_C)[None, None, :] < C))


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load A and B
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
    a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)

    # Perform matmul
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc += tl.dot(a, b)
    acc = acc.to(tl.float16)

    # Apply ReLU if needed
    if ACTIVATION:
        acc = tl.maximum(acc, 0.0)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < M, other=0.0)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load A and B
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
    a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)

    # Perform matmul
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc += tl.dot(a, b)
    acc = acc.to(tl.float16)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < M, other=0.0)


def triton_conv2d(x, w, stride=1):
    batch_size, C_in, H, W = x.shape
    C_out, _, K, _ = w.shape

    # Output dimensions
    out_H = (H - K) // stride + 1
    out_W = (W - K) // stride + 1

    # Create output tensor
    out = torch.empty(batch_size, C_out, out_H, out_W, dtype=torch.float16, device=x.device)

    # Kernel configurations
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C_OUT = 8
    BLOCK_SIZE_C_IN = 8
    TILE_SIZE_H = 16
    TILE_SIZE_W = 16

    # Grid
    grid = lambda meta: (
        triton.cdiv(out_H, meta["BLOCK_SIZE_H"]),
        triton.cdiv(out_W, meta["BLOCK_SIZE_W"]),
        triton.cdiv(C_out, meta["BLOCK_SIZE_C_OUT"]),
        triton.cdiv(C_in, meta["BLOCK_SIZE_C_IN"])
    )

    # Launch kernel
    conv2d_kernel[grid](
        x, w, out,
        H, W, C_in, C_out, K, stride,
        batch_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        TILE_SIZE_H=TILE_SIZE_H,
        TILE_SIZE_W=TILE_SIZE_W
    )
    return out


def triton_max_pool2d(x, stride=2):
    batch_size, C, H, W = x.shape

    # Output dimensions
    out_H = H // stride
    out_W = W // stride

    # Create output tensor
    out = torch.empty(batch_size, C, out_H, out_W, dtype=torch.float16, device=x.device)

    # Grid
    grid = lambda meta: (
        triton.cdiv(out_H, meta["BLOCK_SIZE_H"]),
        triton.cdiv(out_W, meta["BLOCK_SIZE_W"]),
        triton.cdiv(C, meta["BLOCK_SIZE_C"])
    )

    # Kernel configurations
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16

    # Launch kernel
    max_pool2d_kernel[grid](
        x, out,
        H, W, C, stride,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return out


def triton_matmul_relu(a, b, activation=True):
    M, K = a.shape
    K, N = b.shape

    # Create output tensor
    out = torch.empty(M, N, dtype=torch.float16, device=a.device)

    # Autotune
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def kernel(a_ptr, b_ptr, out_ptr, M, N, K,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
               ACTIVATION: tl.constexpr):
        matmul_relu_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                           BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ACTIVATION)

    # Grid
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]))

    # Launch
    kernel[grid](a, b, out, M, N, K, activation=activation)
    return out


def triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape

    # Create output tensor
    out = torch.empty(M, N, dtype=torch.float16, device=a.device)

    # Autotune
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def kernel(a_ptr, b_ptr, out_ptr, M, N, K,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
        matmul_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                      BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # Grid
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]))

    # Launch
    kernel[grid](a, b, out, M, N, K)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Initialize weights in FP16 for Tensor Core usage
        self.conv1_weight = nn.Parameter(torch.randn(6, 1, 5, 5, dtype=torch.float16).cuda())
        self.conv2_weight = nn.Parameter(torch.randn(16, 6, 5, 5, dtype=torch.float16).cuda())
        self.fc1_weight = nn.Parameter(torch.randn(120, 16*5*5, dtype=torch.float16).cuda())
        self.fc2_weight = nn.Parameter(torch.randn(84, 120, dtype=torch.float16).cuda())
        self.fc3_weight = nn.Parameter(torch.randn(num_classes, 84, dtype=torch.float16).cuda())

    def forward(self, x):
        # Conv1 + ReLU + MaxPool
        x = triton_conv2d(x, self.conv1_weight, stride=1)
        x = torch.relu(x)
        x = triton_max_pool2d(x, stride=2)

        # Conv2 + ReLU + MaxPool
        x = triton_conv2d(x, self.conv2_weight, stride=1)
        x = torch.relu(x)
        x = triton_max_pool2d(x, stride=2)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC1 + ReLU
        x = triton_matmul_relu(x, self.fc1_weight, activation=True)

        # FC2 + ReLU
        x = triton_matmul_relu(x, self.fc2_weight, activation=True)

        # FC3
        x = triton_matmul(x, self.fc3_weight)

        return x