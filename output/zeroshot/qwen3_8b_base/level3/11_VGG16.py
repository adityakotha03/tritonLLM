import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
        self.relu_kernel = triton_relu_kernel
        self.matmul_kernel = triton_matmul_kernel
        self.softmax_kernel = triton_softmax_kernel

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

@triton.jit
def triton_relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    out = tl.where(x < 0, 0, x)
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def triton_matmul_kernel(a_ptr, b_ptr, out_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_block = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_id = pid % num_block
    pid = pid // num_block
    block_start = block_id * BLOCK_SIZE
    offs_m = block_start + tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_n[None, :], mask=(offs_m < M)[:, None] & (offs_n < K)[None, :], other=0.0)
    b = tl.load(b_ptr + offs_n[None, :] * K + offs_m[:, None], mask=(offs_n < K)[None, :] & (offs_m < N)[None, :], other=0.0)
    c = tl.dot(a, b)
    tl.store(out_ptr + offs_m, c, mask=offs_m < M)

@triton.jit
def triton_softmax_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(x)
    x -= max_val
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x)
    out = exp_x / sum_exp
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_relu(x):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    triton_relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_matmul(a, b):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    K, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE = 128
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    triton_matmul_kernel[grid](a, b, out, M, N, K, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_softmax(x):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    triton_softmax_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out