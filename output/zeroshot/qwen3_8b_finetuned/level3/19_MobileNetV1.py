import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input (B, O)
    y_ptr,  # Pointer to second input (O,)
    out_ptr,  # Pointer to output (B, O)
    B,  # Batch size
    O,  # Number of output features
    BLOCK_SIZE: tl.constexpr,
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xoffset = xoffset + tl.arange(0, BLOCK_SIZE)
    xmask = xoffset < B * O
    xcol = xoffset % O
    x = tl.load(x_ptr + xoffset, xmask, other=0.0)
    y = tl.load(y_ptr + xcol, xmask, other=0.0)
    tl.store(out_ptr + xoffset, x + y, xmask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the Triton kernel call for bias addition.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    assert x.dim() == 2, "x must be a 2D tensor (B, O)."
    assert y.dim() == 1, "y must be a 1D tensor (O,)."
    assert x.size(1) == y.size(0), "Dimensions must match: x.shape[1] == y.shape[0]."
    B, O = x.size()
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty((B, O), dtype=x.dtype, device=x.device)
    num_elements = B * O
    BLOCK_SIZE = 256
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_kernel[grid](x, y, out, B, O, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture implementation with Triton bias addition.
        """
        super().__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc_weight = nn.Parameter(torch.empty(num_classes, int(1024 * alpha) * 7 * 7))
        self.fc_bias = nn.Parameter(torch.empty(num_classes))
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = x @ self.fc_weight.t()
        x = triton_add(x, self.fc_bias)
        return x