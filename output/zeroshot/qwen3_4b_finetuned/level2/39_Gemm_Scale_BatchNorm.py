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
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = xindex // 16384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 67108864 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tl.device_assert((0 <= x0) & (x0 < 4096) | ~xmask,
        'index out of bounds: 0 <= x0 < 4096')
    tmp7 = tl.load(in_ptr0 + (16384 + x0 + 67108864 * x1), xmask)
    tmp8 = tl.load(in_ptr1 + 1)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tmp7 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tl.device_assert((0 <= x0) & (x0 < 4096) | ~xmask,
        'index out of bounds: 0 <= x0 < 4096')
    tmp14 = tl.load(in_ptr0 + (32768 + x0 + 67108864 * x1), xmask)
    tmp15 = tl.load(in_ptr1 + 2)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tmp14 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tl.device_assert((0 <= x0) & (x0 < 4096) | ~xmask,
        'index out of bounds: 0 <= x0 < 4096')
    tmp21 = tl.load(in_ptr0 + (49152 + x0 + 67108864 * x1), xmask)
    tmp22 = tl.load(in_ptr1 + 3)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp24 = tmp21 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tl.device_assert((0 <= x0) & (x0 < 4096) | ~xmask,
        'index out of bounds: 0 <= x0 < 4096')
    tmp5 = tl.sum(tmp4, 0)[:, None]
    tmp6 = tl.sum(tmp11, 0)[:, None]
    tmp12 = tl.sum(tmp18, 0)[:, None]
    tmp13 = tl.sum(tmp25, 0)[:, None]
    tmp19 = tl.sum(tmp2, 0)[:, None]
    tmp20 = tmp5 + tmp6
    tmp26 = tmp12 + tmp13
    tmp27 = tmp20 + tmp26
    tmp28 = 4.0
    tmp29 = tmp27 / tmp28
    tmp30 = tmp3 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp34 = tl.sum(tmp32, 0)[:, None]
    tmp35 = tmp10 - tmp29
    tmp36 = tmp35 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp39 = tl.sum(tmp37, 0)[:, None]
    tmp40 = tmp17 - tmp29
    tmp41 = tmp40 * tmp40
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.sum(tmp42, 0)[:, None]
    tmp45 = tmp24 - tmp29
    tmp46 = tmp45 * tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp49 = tl.sum(tmp47, 0)[:, None]
    tmp50 = tmp34 + tmp39
    tmp51 = tmp50 + tmp44
    tmp52 = tmp51 + tmp49
    tmp53 = tmp52 / tmp28
    tmp54 = 1e-05
    tmp55 = tmp53 + tmp54
    tmp56 = libdevice.rsqrt(tmp55)
    tl.store(out_ptr0 + x2, tmp29, xmask)
    tl.store(out_ptr1 + x2, tmp56, xmask)
    tl.store(out_ptr2 + x2, tmp56, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1073741824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    x1 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 * tmp9
    tmp10 = tmp7 + tmp8
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (4096, 4096), (4096, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (16384, 4096), (4096, 1))
    assert_size_stride(primals_4, (4096,), (1,))
    assert_size_stride(primals_5, (4096,), (1,))
    assert_size_stride(primals_6, (4096,), (1,))
    assert_size_stride(primals_7, (4096,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (4096, 
            4096), (1, 4096), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(67108864)](buf0, primals_2, buf1, 
            67108864, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((16384, 1), (1, 16384), torch.float32)
        buf3 = empty_strided_cuda((16384, 1), (1, 16384), torch.float32)
        buf4 = empty_strided_cuda((16384, 1), (1, 16384), torch.float32)
        triton_poi_fused_native_batch_norm_1[grid(65536)](buf1, primals_4,
            buf2, buf3, buf4, 65536, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_4
        buf5 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_poi_fused_native_batch_norm_2[grid(1073741824)](buf1,
            primals_3, primals_5, buf2, buf3, primals_6, buf5, 1073741824,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf2
        del buf3
        del primals_6
    return buf5, primals_3, primals_5, primals_7, buf0, buf1, buf4


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_4 = self.scale
        primals_5 = self.bn.weight
        primals_6 = self.bn.bias
        primals_7 = self.bn.weight
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7])
        return output[0]
