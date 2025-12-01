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
def triton_poi_fused_add_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 16384 % 16384
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16384
    x0 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr0 + (x0 + 16384 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + 16384 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = triton_helpers.maximum(tmp4, tmp1)
    tmp6 = tmp0 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tl.store(out_ptr0 + x2, tmp5, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16384
    x0 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr0 + (x0 + 16384 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + 16384 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = triton_helpers.maximum(tmp4, tmp1)
    tmp6 = tmp0 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x1), xmask,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (8192 + x0 + 16384 * x1), xmask,
        eviction_policy='evict_last')
    tmp10 = tmp8 + tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp1)
    tmp12 = tmp0 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tmp7 + tmp13
    tmp15 = tl.load(in_ptr0 + (16384 + x0 + 16384 * x1), xmask,
        eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (16384 + x0 + 16384 * x1), xmask,
        eviction_policy='evict_last')
    tmp17 = tmp15 + tmp16
    tmp18 = triton_helpers.maximum(tmp17, tmp1)
    tmp19 = tmp0 - tmp18
    tmp20 = tl_math.exp(tmp19)
    tmp21 = tmp14 + tmp20
    tl.store(out_ptr0 + x2, tmp21, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16384, 16384), (16384, 1))
    assert_size_stride(primals_2, (16384,), (1,))
    assert_size_stride(primals_3, (128, 16384), (16384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (128, 16384), (1, 
            16384), 0), reinterpret_tensor(primals_1, (16384, 16384), (1, 
            16384), 0), out=buf0)
        del primals_1
        buf1 = buf0
        del buf0
        buf4 = torch.ops.aten.dropout.default(reinterpret_tensor(buf1, (
            128, 16384), (16384, 1), 0), 0.2, False, False, None)
        buf2 = reinterpret_tensor(buf1, (128, 16384), (16384, 1), 0)
        del buf1
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(2097152)](buf2, primals_2, buf4, 
            2097152, XBLOCK=1024, num_warps=8, num_stages=1)
        del buf4
        del primals_2
        buf3 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        buf5 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused__softmax_1[grid(2097152)](buf2, buf3, buf5, buf3,
            2097152, XBLOCK=1024, num_warps=8, num_stages=1)
        buf6 = empty_strided_cuda((128, 16384), (16384, 1), torch.float32)
        triton_poi_fused__softmax_2[grid(2097152)](buf2, buf3, buf6, 
            2097152, XBLOCK=1024, num_warps=8, num_stages=1)
        del buf3
    return buf6, reinterpret_tensor(primals_3, (128, 16384), (1, 16384), 0
        ), buf2


class ModelNew(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.matmul.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]