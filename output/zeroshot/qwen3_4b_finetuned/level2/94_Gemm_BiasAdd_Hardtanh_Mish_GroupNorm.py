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
def triton_poi_fused_add_hardtanh_mish_group_norm_0(in_out_ptr0, in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024 % 8192
    x0 = xindex % 1024
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + x2, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp5
    tmp6 = tmp3 + tmp4
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = triton_helpers.minimum(tmp9, tmp8)
    tmp11 = libdevice.pow(tmp10, tmp10)
    tmp12 = libdevice.exp(tmp4)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 + tmp13
    tl.store(in_out_ptr0 + x3, tmp14, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (1024, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (8192, 8192), (8192, 1))
    assert_size_stride(primals_4, (1024, 8192), (8192, 1))
    assert_size_stride(primals_5, (8192, 256), (256, 1))
    assert_size_stride(primals_6, (256,), (1,))
    assert_size_stride(primals_7, (256,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.mm(primals_4, primals_3)
        assert_size_stride(buf0, (1024, 8192), (8192, 1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardtanh_mish_group_norm_0[grid(1048576)](buf1,
            primals_2, primals_1, primals_5, buf2, 1048576, XBLOCK=512,
            num_warps=8, num_stages=1)
        del primals_2
        del primals_5
        buf3 = empty_strided_cuda((8192, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf2, (8192, 1024), (1024, 1),
            0), reinterpret_tensor(primals_6, (1024, 256), (1, 1024), 0),
            out=buf3)
        buf4 = empty_strided_cuda((1024, 256), (256, 1), torch.float32)
        extern_kernels.addmm(primals_7, reinterpret_tensor(buf2, (1024, 1024
            ), (1024, 1), 0), reinterpret_tensor(primals_6, (1024, 256), (1,
            1024), 0), alpha=1, beta=1, out=buf4)
        del primals_7
    return reinterpret_tensor(buf4, (1024, 8192), (8192, 1), 0
        ), primals_1, primals_3, primals_4, primals_6, buf1, reinterpret_tensor(
        buf2, (8192, 1024), (1024, 1), 0), buf3


class ModelNew(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = self.groupnorm.weight
        primals_5 = self.groupnorm.weight
        primals_4 = self.groupnorm.bias
        primals_6 = self.bias
        primals_7 = self.groupnorm.bias
        primals_8 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8])
        return output[0]
