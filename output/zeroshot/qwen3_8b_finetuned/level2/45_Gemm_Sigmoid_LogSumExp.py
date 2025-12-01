import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_native_layer_norm_sigmoid_0(in_ptr0, in_ptr1,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(out_ptr0 + x2, tmp3, xmask)


@triton.jit
def triton_poi_fused__log_softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 1024 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 1024 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 1024 * x1), xmask)
    tmp7 = tl.load(in_ptr0 + (64 + x0 + 1024 * x1), xmask)
    tmp9 = tl.load(in_ptr0 + (80 + x0 + 1024 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (96 + x0 + 1024 * x1), xmask)
    tmp13 = tl.load(in_ptr0 + (112 + x0 + 1024 * x1), xmask)
    tmp15 = tl.load(in_ptr0 + (128 + x0 + 1024 * x1), xmask)
    tmp17 = tl.load(in_ptr0 + (144 + x0 + 1024 * x1), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp19 = triton_helpers.maximum(tmp18, tmp1)
    tmp20 = tmp0 - tmp19
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp2 - tmp19
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp4 - tmp19
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp6 - tmp19
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp8 - tmp19
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp30 + tmp32
    tmp34 = tmp10 - tmp19
    tmp35 = tl_math.exp(tmp34)
    tmp36 = tmp33 + tmp35
    tmp37 = tmp12 - tmp19
    tmp38 = tl_math.exp(tmp37)
    tmp39 = tmp36 + tmp38
    tmp40 = tmp14 - tmp19
    tmp41 = tl_math.exp(tmp40)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp16 - tmp19
    tmp44 = tl_math.exp(tmp43)
    tmp45 = tmp42 + tmp44
    tmp46 = tmp18 - tmp19
    tmp47 = tl_math.exp(tmp46)
    tmp48 = tmp45 + tmp47
    tmp49 = tl_math.log(tmp48)
    tl.store(out_ptr0 + (x0 + 1024 * x1), tmp49, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4096, 2048), (2048, 1))
    assert_size_stride(primals_2, (4096,), (1,))
    assert_size_stride(primals_3, (16384, 2048), (2048, 1))
    assert_size_stride(primals_4, (1024, 4096), (4096, 1))
    assert_size_stride(primals_5, (1024,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (2048, 
            4096), (1, 2048), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((16384, 1024), (1024, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_sigmoid_0[grid(16384)](buf0,
            primals_2, buf1, 16384, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((16384, 1024), (1024, 1), torch.float32)
        extern_kernels.addmm(primals_5, buf1, reinterpret_tensor(primals_4,
            (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf2)
        del primals_5
        buf3 = empty_strided_cuda((16384,), (1,), torch.float32)
        triton_poi_fused__log_softmax_1[grid(16384)](buf2, buf3, 16384,
            XBLOCK=16, num_warps=4, num_stages=1)
        del buf2
    return buf3, primals_3, buf0, buf1, primals_4, buf3


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_0):
        primals_1 = self.linear1.weight
        primals_2 = self.linear1.bias
        primals_4 = self.linear2.weight
        primals_5 = self.linear2.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]