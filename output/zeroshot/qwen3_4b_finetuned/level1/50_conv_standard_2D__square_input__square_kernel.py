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


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 322560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 576 % 96
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3 % 576), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (x3 % 576), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp1
    tmp6 = tmp2 + tmp4
    tmp7 = tmp5 + tmp3
    tmp8 = tmp7 + tmp6
    tl.store(out_ptr0 + x3, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (96, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(primals_2, (96,), (1,))
    assert_size_stride(primals_3, (256, 3, 224, 224), (150528, 50176, 224,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(4, 
            4), padding=(2, 2), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (256, 96, 56, 56), (295680, 3072, 56, 1))
        buf1 = empty_strided_cuda((256, 96, 576), (55296, 576, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(322560)](buf0, primals_2, primals_1,
            primals_2, primals_1, buf1, 322560, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_1
        del primals_2
    return buf1, primals_3, buf0


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
            stride=4, padding=2)
    
    def forward(self, input_0):
        primals_1 = self.conv1.weight
        primals_2 = self.conv1.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
