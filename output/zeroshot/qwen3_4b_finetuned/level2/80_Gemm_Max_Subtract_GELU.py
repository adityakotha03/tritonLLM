import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 8192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 8192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 - tmp14
    tmp16 = tmp1 - tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp3 - tmp14
    tmp19 = tmp17 + tmp18
    tmp20 = tmp5 - tmp14
    tmp21 = tmp19 + tmp20
    tmp22 = tmp7 - tmp14
    tmp23 = tmp21 + tmp22
    tmp24 = tmp9 - tmp14
    tmp25 = tmp23 + tmp24
    tmp26 = tmp11 - tmp14
    tmp27 = tmp25 + tmp26
    tmp28 = tmp13 - tmp14
    tmp29 = tmp27 + tmp28
    tmp30 = 0.01
    tmp31 = tmp29 * tmp30
    tmp32 = 0.001
    tmp33 = tmp29 * tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp31 + tmp34
    tl.store(out_ptr0 + x0, tmp35, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.addmm(primals_2, primals_3, tl.broadcast_to(primals_1,
            (8192, 8192)), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((1024, 1, 8192), (8192, 8192, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused_max_sub_0[grid(1024)](buf0, buf1, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
    return buf1, primals_3, buf0


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
