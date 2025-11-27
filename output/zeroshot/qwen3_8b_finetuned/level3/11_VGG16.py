import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    in_out_ptr,  # Pointer to input (and output)
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Elementwise ReLU (max(0, x)) for a contiguous tensor."""
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = xoffset < n_elements
    x = tl.load(in_out_ptr + xoffset, mask, other=0.0)
    y = tl.full([1], 0.0, tl.int32)
    tl.store(in_out_ptr + xoffset, tl.maximum(x, y), mask)


def triton_relu(x: torch.Tensor):
    """
    Perform ReLU on a contiguous tensor using a Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable, power-of-two block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # VGG16 architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            triton_relu,  # Replaced ReLU with Triton kernel
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            triton_relu,  # Replaced ReLU with Triton kernel
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x