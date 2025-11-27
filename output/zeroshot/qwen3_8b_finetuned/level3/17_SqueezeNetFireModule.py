import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_relu_0(in_out_ptr0, n_elements, XBLOCK: tl.constexpr):
    xnumel = n_elements
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp0 * tmp2
    tl.store(in_out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, n_elements, XBLOCK: tl.constexpr):
    xnumel = n_elements
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp0 * tmp2
    tl.store(in_out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_relu_2(in_out_ptr0, n_elements, XBLOCK: tl.constexpr):
    xnumel = n_elements
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp0 * tmp2
    tl.store(in_out_ptr0 + x0, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 3, 256, 256), (3*256*256, 256*256, 256, 1))
    assert_size_stride(primals_2, (6, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_3, (6, 6, 1, 1), (6, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 6, 256, 256), (6*256*256, 256*256, 256, 1),
            torch.float32)
        buf1 = buf0
        del buf0
        get_raw_buf = buf1
        buf2 = buf1
        del buf1
        triton_poi_fused_relu_0[grid(2359296)](get_raw_buf, 2359296, XBLOCK=128,
            num_warps=4, num_stages=1)
        buf3 = buf2
        del buf2
        buf4 = buf3
        del buf3
        triton_poi_fused_relu_1[grid(2359296)](buf4, 2359296, XBLOCK=128,
            num_warps=4, num_stages=1)
        buf5 = buf4
        del buf4
        buf6 = buf5
        del buf5
        triton_poi_fused_relu_2[grid(2359296)](buf6, 2359296, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf6
    return primals_1, primals_2, primals_3, buf0, buf2, buf4, buf5, buf1


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels,
        expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = triton_poi_fused_relu_0
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels,
            kernel_size=1)
        self.expand1x1_activation = triton_poi_fused_relu_1
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels,
            kernel_size=3, padding=1)
        self.expand3x3_activation = triton_poi_fused_relu_2
    
    def forward(self, input_0):
        primals_1 = input_0
        primals_2 = self.squeeze.weight
        primals_3 = self.expand1x1.weight
        primals_4 = self.expand3x3.weight
        output = call([primals_1, primals_2, primals_3])
        return output[7]