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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_avg_pool2d_gelu_mul_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8192
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 8192 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 8192 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 8192 * x2), xmask)
    tmp7 = tl.load(in_ptr0 + (64 + x0 + 8192 * x2), xmask)
    tmp9 = tl.load(in_ptr0 + (80 + x0 + 8192 * x2), xmask)
    tmp11 = tl.load(in_ptr0 + (96 + x0 + 8192 * x2), xmask)
    tmp13 = tl.load(in_ptr0 + (112 + x0 + 8192 * x2), xmask)
    tmp15 = tl.load(in_ptr0 + (128 + x0 + 8192 * x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 16.0
    tmp18 = tmp16 / tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = libdevice.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 2.0
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr0 + x3, tmp28, xmask)


@triton.jit
def triton_poi_fused_max_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 8192 + x0, xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tl.load(in_ptr0 + 16384 + x0, xmask)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.load(in_ptr0 + 24576 + x0, xmask)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tl.load(in_ptr0 + 32768 + x0, xmask)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.load(in_ptr0 + 40960 + x0, xmask)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tl.load(in_ptr0 + 49152 + x0, xmask)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp13 = tl.load(in_ptr0 + 57344 + x0, xmask)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp15 = tl.load(in_ptr0 + 65536 + x0, xmask)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tl.load(in_ptr0 + 73728 + x0, xmask)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tl.load(in_ptr0 + 81920 + x0, xmask)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tl.store(out_ptr0 + x0, tmp20, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(primals_2, reinterpret_tensor(primals_1, (8192, 
            8192), (1, 8192), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((1024, 8192, 16), (131072, 16, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_gelu_mul_0[grid(8388608)](buf0, buf1, 
            8388608, XBLOCK=256, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((1024, 16), (16, 1), torch.float32)
        triton_poi_fused_max_1[grid(1024)](buf1, buf2, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
    return buf2, primals_2, buf0, buf1


class ModelNew(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = self.scale_factor
        primals_4 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]