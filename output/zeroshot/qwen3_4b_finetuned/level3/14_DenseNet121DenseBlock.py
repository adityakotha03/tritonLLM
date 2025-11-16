import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 110592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_relu_threshold_backward_2(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16 % 64
    x0 = xindex % 16
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x1 + 512 * x2), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x1 + 512 * x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x1 + 512 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + x4, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (6, 64, 64, 64), (4096, 64, 1, 1))
    assert_size_stride(primals_2, (6, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_3, (6, 64), (64, 1))
    assert_size_stride(primals_4, (10, 32, 224, 224), (154240, 224, 224, 1))
    assert_size_stride(primals_5, (6,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 1, 224, 224), (512, 512, 1, 1), torch
            .float32)
        get_raw_stream(0)
        triton_poi_fused_0[grid(221184)](primals_4, buf0, 221184, XBLOCK=
            256, num_warps=4, num_stages=1)
        del primals_4
        buf1 = empty_strided_cuda((6, 576, 1, 1), (576, 1, 1, 1), torch.float32)
        triton_poi_fused_1[grid(110592)](primals_2, buf1, 110592, XBLOCK=
            512, num_warps=4, num_stages=1)
        del primals_2
        buf2 = extern_kernels.batch_norm(primals_1, buf0, None, None, True)
        del primals_1
        buf3 = empty_strided_cuda((10, 64, 224, 224), (327680, 5120, 2, 1),
            torch.float32)
        buf7 = empty_strided_cuda((10, 64, 224, 224), (327680, 5120, 2, 1),
            torch.bool)
        triton_poi_fused_relu_threshold_backward_2[grid(221184)](buf3,
            buf2, buf7, 221184, XBLOCK=256, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((10, 576, 224, 224), (25165824, 43680, 1,
            1), torch.float32)
        extern_kernels.convolution(buf3, buf1, stride=(1, 1), padding=(1, 
            1), dilation=(1, 1), transposed=False, output_padding=(0, 0),
            groups=1, bias=None)
        buf5 = empty_strided_cuda((10, 576, 224, 224), (25165824, 43680, 1,
            1), torch.float32)
        triton_poi_fused_convolution_3[grid(4096)](buf4, buf5, 4096, XBLOCK
            =128, num_warps=4, num_stages=1)
        del buf4
        buf6 = extern_kernels.batch_norm(primals_3, buf5, None, None, True)
        del primals_3
    return buf6, reinterpret_tensor(buf3, (10, 576, 224, 224), (327680, 1,
        224, 1), 0), reinterpret_tensor(buf0, (64, 224, 224), (5120, 224, 1
        ), 0), reinterpret_tensor(buf2, (64, 1, 224, 224), (5120, 1, 1, 1),
        0), buf5, reinterpret_tensor(buf1, (6, 576, 1, 1), (576, 1, 1, 1), 0
        ), buf7


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )
    
    def forward(self, input_0):
        primals_1 = self.layers[0][0].weight
        primals_2 = self.layers[0][2].weight
        primals_3 = self.layers[0][1].weight
        primals_4 = input_0
        primals_5 = self.layers[0][3].weight
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
