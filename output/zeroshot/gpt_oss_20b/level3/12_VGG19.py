import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------- Triton kernels ----------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_relu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M, N, K,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    row_start = pid_x * BLOCK_SIZE
    col_start = pid_y * BLOCK_SIZE

    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE):
        a_tile = tl.load(
            a_ptr + row_offsets[:, None] * K + (k + tl.arange(0, BLOCK_SIZE)[None, :]),
            mask=row_offsets[:, None] < M,
            other=0.0,
        ).to(tl.float32)

        b_tile = tl.load(
            b_ptr + (k + tl.arange(0, BLOCK_SIZE)[:, None]) * N + col_offsets[None, :],
            mask=col_offsets[None, :] < N,
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a_tile, b_tile)

    # ReLU
    acc = tl.where(acc > 0, acc, 0)

    # Store
    mask_rows = row_offsets[:, None] < M
    mask_cols = col_offsets[None, :] < N
    mask = mask_rows & mask_cols
    tl.store(out_ptr + row_offsets[:, None] * N + col_offsets[None, :], acc, mask=mask)


# Wrapper functions
def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    """
    a: (M, K)  float32
    b: (K, N)  float32
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match."

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE = 128
    grid = lambda meta: (
        (M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    matmul_relu_kernel[grid](
        a, b, out, M, N, K, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# ---------------- Model definition ----------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Features (same as original VGG19)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Replace linear layers with Triton matmul + ReLU
        self.weight1 = nn.Parameter(torch.empty((512 * 7 * 7, 4096), device="cuda", dtype=torch.float32))
        self.bias1 = nn.Parameter(torch.empty(4096, device="cuda", dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.zeros_(self.bias1)

        self.weight2 = nn.Parameter(torch.empty((4096, 4096), device="cuda", dtype=torch.float32))
        self.bias2 = nn.Parameter(torch.empty(4096, device="cuda", dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        nn.init.zeros_(self.bias2)

        self.weight3 = nn.Parameter(torch.empty((4096, num_classes), device="cuda", dtype=torch.float32))
        self.bias3 = nn.Parameter(torch.empty(num_classes, device="cuda", dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight3, a=math.sqrt(5))
        nn.init.zeros_(self.bias3)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        # First linear + ReLU
        x = triton_matmul_relu(x, self.weight1)
        x = x + self.bias1

        # Second linear + ReLU
        x = triton_matmul_relu(x, self.weight2)
        x = x + self.bias2

        # Final linear
        x = triton_matmul_relu(x, self.weight3)
        x = x + self.bias3
        return x