import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_div_mul_rsub_sub_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1887904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    x3 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + (32 + x0 + 64 * x3), xmask)
    tmp2 = tl.load(in_ptr0 + (64 + x0 + 64 * x3), xmask)
    tmp3 = tl.load(in_ptr0 + (96 + x0 + 64 * x3), xmask)
    tmp4 = tmp0 + tmp1
    tmp5 = tmp2 + tmp4
    tmp6 = tmp3 + tmp5
    tmp7 = tmp0 * tmp0
    tmp8 = tmp1 * tmp1
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 * tmp2
    tmp11 = tmp9 + tmp10
    tmp12 = tmp3 * tmp3
    tmp13 = tmp11 + tmp12
    tmp14 = 4.0
    tmp15 = tmp6 / tmp14
    tmp16 = tmp13 / tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp0 - tmp15
    tmp22 = tmp21 * tmp20
    tmp23 = tmp1 - tmp15
    tmp24 = tmp23 * tmp20
    tmp25 = tmp22 + tmp24
    tmp26 = tmp2 - tmp15
    tmp27 = tmp26 * tmp20
    tmp28 = tmp25 + tmp27
    tmp29 = tmp3 - tmp15
    tmp30 = tmp29 * tmp20
    tmp31 = tmp28 + tmp30
    tl.store(out_ptr0 + x2, tmp31, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (112, 64, 512, 512), (2097152, 32768, 64, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 64, 512, 512), (2097152, 32768, 64, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_div_mul_rsub_sub_0[grid(1887904)](primals_1,
            buf0, 1887904, XBLOCK=256, num_warps=4, num_stages=1)
        buf1 = extern_kernels.addmm(primals_2, primals_1, primals_1, alpha=
            1, beta=1)
        del primals_2
    return buf1, primals_1, buf0


class ModelNew(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, input_0):
        primals_2 = self.inorm.weight
        primals_1 = self.inorm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]