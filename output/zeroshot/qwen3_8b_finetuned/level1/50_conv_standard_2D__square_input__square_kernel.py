import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1353977536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 96
    x1 = xindex // 96
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 96 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr2 + x3, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (96, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (96,), (1,))
    assert_size_stride(primals_3, (256, 3, 224, 224), (150528, 50176, 224, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (256, 96, 224, 224), (4703936, 48931, 216,
            1))
        buf1 = empty_strided_cuda((256, 96, 224, 224), (4703936, 48931, 216,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1353977536)](primals_2,
            primals_1, primals_3, buf1, 1353977536, XBLOCK=1024,
            num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        del primals_3
    return buf1, buf0


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, input_0):
        primals_1 = self.conv1.weight
        primals_2 = self.conv1.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]