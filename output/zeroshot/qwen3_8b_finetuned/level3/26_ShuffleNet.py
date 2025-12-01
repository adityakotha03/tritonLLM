1. High‑level goal  
Replace the element‑wise addition that appears in the shortcut branch of each ShuffleNet unit with a Triton kernel. The original PyTorch code performs `out += self.shortcut(x)`. The Triton kernel must compute the same element‑wise sum but with the additional benefit of a fused‑memory‑access pattern that respects the 2‑D layout of the tensors (batch × channel × height × width).

2. Tensor shapes and indexing  
- Input tensors `x` and `shortcut_out` are both 4‑D: `(B, C, H, W)`.  
- The kernel flattens the 4‑D view to a 1‑D linear index `i = batch*CHW + channel*HW + height*W + width`.  
- The linear index is split into two dimensions that correspond to the original layout: `x2 = i % (C*H*W)` (the channel‑wise offset) and `x0 = i // (C*H*W)` (the batch offset). This split reproduces the original `(batch, channel, height, width)` ordering used by PyTorch’s `+` operator.  
- The address calculation `x0 + 64*x2` (where `64 = C*H*W` for the test case) yields the correct pointer arithmetic because each batch stride is `C*H*W` and each channel stride is `H*W`. This matches the stride pattern asserted by `assert_size_stride`.

3. Parallelization & launch configuration  
- **Program ID axis**: Only one axis (`program_id(0)`) is needed because the total number of elements (`n_elements`) fits into a 1‑D grid.  
- **BLOCK_SIZE**: Chosen as 256 (a power‑of‑two that divides the total element count evenly for the example). This value is tuned by the autotuner; the kernel is generic and works for any `BLOCK_SIZE`.  
- **Grid**: Computed as `ceil(n_elements / BLOCK_SIZE)`. The lambda `grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)` implements this.  
- **Warps & stages**: `num_warps=4` and `num_stages=1` are the default heuristics from the generated skeleton; they give enough parallelism for 256 threads per block while keeping shared‑memory pressure low.

4. Memory access pattern  
- **Loads**: Each thread loads a single element from each input using the linear index (`x0 + 64*x2`). The `mask` ensures that threads whose linear index exceeds `n_elements` do not read/write.  
- **Coalescing**: Because consecutive threads in a warp differ only in the last dimension (`x2`), their memory accesses stride by 1 across the innermost contiguous dimension, achieving full coalescing. The batch dimension (`x0`) varies by a stride of `C*H*W`, which is a multiple of the warp size, so each warp reads a contiguous chunk of 64 elements from each tensor.  
- **Stores**: The result is written back with the same linear index, preserving the original layout. No intermediate transposition or reshaping is needed because the kernel works on the flattened view that matches PyTorch’s default memory layout.

5. Numerics & correctness details  
- **Data type**: The kernel operates on `float32` (the default dtype of the model). No casting is required; the Triton `tl.load` and `tl.store` default to the pointer’s dtype.  
- **Broadcasting**: The addition is element‑wise; no broadcasting is performed because both inputs share the exact shape. If a broadcast were needed, the kernel would need extra logic to compute the broadcasted index, but that is not the case here.  
- **Edge handling**: The mask (`offsets < n_elements`) guarantees that the final partial block does not read/write out‑of‑bounds. This is crucial when `n_elements` is not a multiple of `BLOCK_SIZE`.  
- **Determinism**: All operations are deterministic: addition is performed in a single warp, no reduction or atomics are required, and the kernel does not rely on nondeterministic memory ordering.

6. Summary checklist for the Triton kernel  
- **Shape verification**: `assert_size_stride` ensures the input tensors have the expected contiguous layout `(B, C, H, W)` with strides `(C*H*W, H*W, W, 1)`.  
- **Index decomposition**: `x2 = i % (C*H*W)` and `x0 = i // (C*H*W)` reproduce the original batch‑channel‑height‑width ordering.  
- **Block size**: Choose a power‑of‑two (`BLOCK_SIZE`) that divides the total element count evenly for maximal occupancy; autotuning selects the optimal value.  
- **Grid computation**: `grid = ceil(n_elements / BLOCK_SIZE)` guarantees full coverage.  
- **Memory accesses**: Load both inputs with a stride‑1 pattern across the innermost dimension; store the result with the same stride.  
- **Masking**: Apply a boolean mask to the load/store to guard against out‑of‑bounds threads.  
- **Numerics**: Use the same dtype as the inputs; no extra casts or precision loss.  
- **Integration**: Replace the PyTorch `+` in the shortcut branch with a call to `triton_add` (or the fused kernel). The surrounding `nn.Module` remains unchanged; only the element‑wise addition is replaced.

Following these reasoning steps yields a Triton kernel that faithfully reproduces the PyTorch shortcut addition while exploiting the GPU’s memory hierarchy and parallel execution model.</think>

```python
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 153600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex // 64 % 48
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (1, 24, 224, 224), (114688, 4778, 22, 1))
    assert_size_stride(primals_2, (24, 3, 3, 1), (27, 9, 3, 1))
    assert_size_stride(primals_3, (24,), (1,))
    assert_size_stride(primals_4, (24,), (1,))
    assert_size_stride(primals_5, (1, 24, 224, 224), (114688, 4778, 22, 1))
    assert_size_stride(primals_6, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_7, (24,), (1,))
    assert_size_stride(primals_8, (24,), (1,))
    assert_size_stride(primals_9, (1, 24, 224, 224), (114688, 4778, 22, 1))
    assert_size_stride(primals_10, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_11, (24,), (1,))
    assert_size_stride(primals_12, (24,), (1,))
    assert_size_stride(primals_13, (1024, 24, 1, 1), (24, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 24, 224, 224), (114688, 4778, 22, 1))
        buf1 = extern_kernels.convolution(primals_5, primals_6, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf1, (1, 24, 224, 224), (114688, 4778, 22, 1))
        buf2 = extern_kernels.convolution(primals_9, primals_10, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf2, (1, 24, 224, 224), (114688, 4778, 22, 1))
        buf3 = extern_kernels.convolution(buf0, primals_13, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (1, 1024, 224, 224), (50331648, 4896, 24, 1))
        buf4 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024),
            torch.float32)
        extern_kernels.addmm(primals_11, buf3, reinterpret_tensor(primals_12,
            (1024, 1), (1, 1024), 0), alpha=1, beta=1, out=buf4)
        del primals_12
        buf5 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        buf6 = empty_strided_cuda((1, 24, 224, 224), (114688, 4778, 22, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(153600)](buf5, primals_1, buf6, 153600,
            XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        buf7 = extern_kernels.convolution(buf6, primals_6, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=24, bias=None)
        buf8 = extern_kernels.convolution(primals_5, primals_6, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=24, bias=None)
        buf9 = empty_strided_cuda((1, 24, 224, 224), (114688, 4778, 22, 1),
            torch.float32)
        triton_poi_fused_add_0[grid(153600)](buf7, buf8, buf9, 153600,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf8
        buf10 = extern_kernels.convolution(buf9, primals_10, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=24, bias=None)
        buf11 = extern_kernels.convolution(primals_9, primals_10, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=24, bias=None)
        buf12 = empty_strided_cuda((1, 24, 224, 224), (114688, 4778, 22, 1),
            torch.float32)
        triton_poi_fused_add_0[grid(153600)](buf10, buf11, buf12, 153600,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf11
        buf13 = extern_kernels.convolution(buf12, primals_13, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        buf14 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024),
            torch.float32)
        extern_kernels.addmm(primals_11, buf13, reinterpret_tensor(primals_12,
            (1024, 1), (1, 1024), 0), alpha=1, beta=1, out=buf14)
        del primals_12
        buf15 = extern_kernels.convolution(buf12, primals_13, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        buf16 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024),
            torch.float32)
        extern_kernels.addmm(primals_11, buf15, reinterpret_tensor(primals_12,
            (1024, 1), (1, 1024), 0), alpha=1, beta=1, out=buf16)
        del primals_12
    return buf16, primals_2, primals_3, primals_4, primals_6, primals_7, primals_8, primals_10, primals_11, primals_13, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf9, buf10, buf12, buf13, buf15


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose
        x = x.transpose(1, 2).contiguous()
        
        # Flatten
        x = x.view(batch_size, -1, height, width)
        
        return x


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ShuffleNetUnit, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        """
        ShuffleNet architecture.

        :param num_classes: Number of output classes.
        :param groups: Number of groups for group convolution.
        :param stages_repeats: List of ints specifying the number of repeats for each stage.
        :param stages_out_channels: List of ints specifying the output channels for each stage.
        """
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        """
        Helper function to create a stage of ShuffleNet units.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param repeats: Number of ShuffleNet units in the stage.
        :param groups: Number of groups for group convolution.
        :return: nn.Sequential containing the stage.
        """
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, input_0):
        primals_2 = self.conv1.weight
        primals_3 = self.bn1.weight
        primals_4 = self.bn1.bias
        primals_13 = self.conv5.weight
        primals_11 = self.bn5.weight
        primals_12 = self.bn5.bias
        primals_1 = input_0
        primals_5 = self.stage2[0].conv1.weight
        primals_7 = self.stage2[0].bn1.weight
        primals_8 = self.stage2[0].bn1.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_6 = self.stage2[0].bn2.weight
        primals_9 = self.stage2[0].bn2.bias
        primals_10 = self.stage2[0].conv2.weight
        primals_