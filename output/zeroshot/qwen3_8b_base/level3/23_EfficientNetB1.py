import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        self.fc = nn.Linear(1280, num_classes)
        
        # Triton kernels
        self._register_triton_kernels()
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def _register_triton_kernels(self):
        self._register_relu6_kernel()
        self._register_add_kernel()
        self._register_avg_pool2d_kernel()
        self._register_linear_kernel()
    
    @triton.jit
    def _relu6_kernel(
        x_ptr, 
        out_ptr, 
        n_elements, 
        BLOCK_SIZE: tl.constexpr
    ):
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = tl.where(x > 6.0, 6.0, x)
        tl.store(out_ptr + offsets, out, mask=mask)
    
    @triton.jit
    def _add_kernel(
        x_ptr, 
        y_ptr, 
        out_ptr, 
        n_elements, 
        BLOCK_SIZE: tl.constexpr
    ):
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        out = x + y
        tl.store(out_ptr + offsets, out, mask=mask)
    
    @triton.jit
    def _avg_pool2d_kernel(
        x_ptr, 
        out_ptr, 
        height, 
        width, 
        kernel_size, 
        stride, 
        padding, 
        n_channels, 
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start_h = pid * BLOCK_SIZE
        block_start_w = 0
        offset_h = block_start_h + tl.arange(0, BLOCK_SIZE)
        offset_w = block_start_w + tl.arange(0, BLOCK_SIZE)
        mask_h = offset_h < height
        mask_w = offset_w < width
        mask = mask_h & mask_w
        
        x = tl.load(x_ptr + offset_h[:, None] * width + offset_w[None, :], mask=mask, other=0.0)
        x = tl.sum(x, axis=(0, 1))
        out = x / (kernel_size * kernel_size)
        tl.store(out_ptr + pid, out, mask=mask)
    
    @triton.jit
    def _linear_kernel(
        x_ptr, 
        weight_ptr, 
        bias_ptr, 
        out_ptr, 
        n_elements, 
        BLOCK_SIZE: tl.constexpr
    ):
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = x * weight + bias
        tl.store(out_ptr + offsets, out, mask=mask)
    
    def _register_relu6_kernel(self):
        def relu6(x: torch.Tensor):
            return self._relu6_kernel(x, x, x.numel(), 128)
        self.relu6 = relu6
    
    def _register_add_kernel(self):
        def add(x: torch.Tensor, y: torch.Tensor):
            out = torch.empty_like(x)
            grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
            self._add_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=128)
            return out
        self.add = add
    
    def _register_avg_pool2d_kernel(self):
        def avg_pool2d(x: torch.Tensor, kernel_size=1, stride=1, padding=0):
            height, width = x.shape[2], x.shape[3]
            out_channels = x.shape[1]
            out = torch.empty((out_channels, height, width), device=x.device)
            grid = lambda meta: (out.shape[1],)
            self._avg_pool2d_kernel[grid](x, out, height, width, kernel_size, stride, padding, out_channels, BLOCK_SIZE=128)
            return out
        self.avg_pool2d = avg_pool2d
    
    def _register_linear_kernel(self):
        def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
            out = torch.empty_like(x)
            grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
            self._linear_kernel[grid](x, weight, bias, out, x.numel(), BLOCK_SIZE=128)
            return out
        self.linear = linear
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = self._relu6(self.bn2(self.conv2(x)))
        x = self.avg_pool2d(x, kernel_size=1, stride=1, padding=0)
        x = torch.flatten(x, 1)
        x = self.linear(x, self.fc.weight, self.fc.bias)
        
        return x