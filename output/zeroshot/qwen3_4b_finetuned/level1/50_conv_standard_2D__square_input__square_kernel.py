import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 6969600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 57600 % 96
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (96, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(primals_2, (96,), (1,))
    assert_size_stride(primals_3, (256, 3, 224, 224), (150528, 50176, 224, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [0,
            0, 0, 0], stride=(4, 4), padding=(2, 2), dilation=(1, 1),
            transposed=False, output_padding=(0, 0), groups=1, bias=None)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(6969600)](buf2, primals_2, 
            6969600, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
    return buf2, primals_1, primals_3


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
