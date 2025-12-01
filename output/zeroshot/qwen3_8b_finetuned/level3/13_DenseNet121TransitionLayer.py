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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_relu_0(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2064384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 65536
    x0 = xindex % 65536
    x4 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + x4, xmask, eviction_policy='evict_last')
    tmp4 = tmp0 - tmp1
    tmp5 = 1e-05
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0 / tmp6
    tmp8 = tmp4 * tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tmp10 + tmp3
    tmp12 = tl.full([1], 4, tl.int32)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp12
    tmp15 = tmp14 * tmp12
    tmp16 = tmp15 * tmp12
    tmp17 = tmp16 * tmp12
    tmp18 = tmp17 / tmp13
    tmp19 = -inf
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tl.store(out_ptr0 + x3, tmp11, xmask)
    tl.store(out_ptr1 + x3, tmp20, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (128, 32, 256, 256), (2097152, 65536, 256,
        1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (32,), (1,))
    assert_size_stride(primals_4, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32, 256, 256), (2097152, 65536, 256,
            1), torch.float32)
        buf1 = empty_strided_cuda((128, 32, 256, 256), (2097152, 65536, 256,
            1), torch.float32)
        buf2 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        buf3 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_convolution_relu_0[grid(2064384)](buf0,
            primals_2, primals_3, primals_4, buf2, buf3, 2064384, XBLOCK=128,
            num_warps=4, num_stages=1)
        del primals_2
        del primals_3
        del primals_4
        buf4 = extern_kernels.convolution(reinterpret_tensor(buf2, (128, 64,
            256, 256), (4194304, 0, 16384, 65536), 0), primals_1, stride=(1,
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (128, 64, 256, 256), (4194304, 65536, 256, 
            1))
        buf5 = extern_kernels.convolution(buf4, primals_1, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (128, 64, 128, 128), (1048576, 16384, 128, 
            1))
    return buf5, reinterpret_tensor(buf2, (128, 64, 256, 256), (4194304, 0,
        16384, 65536), 0), buf0, buf1, buf3, primals_1


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input_0):
        primals_1 = self.transition[2].weight
        primals_2 = self.transition[0].weight
        primals_3 = self.transition[0].bias
        primals_4 = self.transition[3].weight
        primals_1 = primals_1
        primals_3 = primals_3
        primals_2 = primals_2
        primals_4 = primals_4
        output = call([input_0, primals_1, primals_2, primals_3, primals_4])
        return output[0]