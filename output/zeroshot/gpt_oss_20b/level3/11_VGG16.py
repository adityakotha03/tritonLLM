import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------- Triton kernels ---------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16},
                      num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32},
                      num_stages=3, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def linear_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M, N, K,
    stride_input: tl.constexpr,
    stride_weight: tl.constexpr,
    stride_output: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    # grid: (ceil(M/BLOCK_M),)
    row = pid * BLOCK_M
    col = tl.arange(0, BLOCK_N)

    row_mask = row + tl.arange(0, BLOCK_M) < M
    col_mask = col < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load tiles from input and weight
        input_tile = tl.load(
            input_ptr + (row[:, None] * stride_input + k[None, :] * stride_input),
            mask=row_mask[:, None] & (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        weight_tile = tl.load(
            weight_ptr + (k[:, None] * stride_weight + col[None, :] * stride_weight),
            mask=(k[:, None] < K) & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(input_tile, weight_tile)

    # add bias and apply ReLU
    bias = tl.load(bias_ptr + col, mask=col_mask, other=0.0).to(tl.float32)
    acc = acc + bias
    acc = tl.maximum(acc, 0.0)

    # store result
    tl.store(
        output_ptr + (row[:, None] * stride_output + col[None, :] * stride_output),
        acc.to(tl.float16),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def triton_linear_relu(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Custom Triton implementation of Linear layer with bias addition and ReLU activation.
    input: (batch, in_features)
    weight: (in_features, out_features)
    bias: (out_features,)
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda
    batch, in_f = input.shape
    out_f = weight.shape[1]

    output = torch.empty((batch, out_f), dtype=torch.float16, device=input.device)

    BLOCK_M = 128  # will be autotuned
    BLOCK_N = 128

    grid = lambda meta: (triton.cdiv(batch, meta["BLOCK_M"]),)

    linear_relu_kernel[grid](
        input,
        weight,
        bias,
        output,
        batch,
        out_f,
        in_f,
        input.stride(0),
        weight.stride(0),
        output.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return output.float()


# --------------------------- Model definition ---------------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Same convolutional backbone as VGG16
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Replacing the fully‑connected head with Triton kernels
        self.fc1_weight = nn.Parameter(torch.randn(512 * 7 * 7, 4096).half())
        self.fc1_bias = nn.Parameter(torch.randn(4096).half())
        self.fc2_weight = nn.Parameter(torch.randn(4096, 4096).half())
        self.fc2_bias = nn.Parameter(torch.randn(4096).half())
        self.fc3_weight = nn.Parameter(torch.randn(4096, num_classes).half())
        self.fc3_bias = nn.Parameter(torch.randn(num_classes).half())

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1).to(torch.float16)

        # First fully‑connected layer + ReLU
        x = triton_linear_relu(x, self.fc1_weight, self.fc1_bias)
        # Second fully‑connected layer + ReLU
        x = triton_linear_relu(x, self.fc2_weight, self.fc2_bias)
        # Final output layer (no activation)
        x = triton_linear_relu(x, self.fc3_weight, self.fc3_bias)
        return x