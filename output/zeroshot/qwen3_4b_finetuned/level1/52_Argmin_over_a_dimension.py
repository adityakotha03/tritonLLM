import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_argmin_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x1), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp8 != 0
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp8 == 0
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tl.where(tmp9, tmp8, tmp13)
    tmp15 = tmp3 > tmp2
    tmp16 = tmp15.to(tl.int64)
    tmp17 = tmp16 != 0
    tmp18 = tl.full([1], 2, tl.int64)
    tmp19 = tmp16 == 0
    tmp20 = tl.where(tmp19, tmp12, tmp18)
    tmp21 = tl.where(tmp17, tmp16, tmp20)
    tmp22 = tl.where(tmp14, tmp14, tmp21)
    tmp23 = tmp5 > tmp4
    tmp24 = tmp23.to(tl.int64)
    tmp25 = tmp24 != 0
    tmp26 = tl.full([1], 3, tl.int64)
    tmp27 = tmp24 == 0
    tmp28 = tl.where(tmp27, tmp12, tmp26)
    tmp29 = tl.where(tmp22, tmp22, tmp28)
    tl.store(out_ptr0 + x2, tmp29, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096, 4095), (16384, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4096), (4096, 1), torch.int64)
        get_raw_stream(0)
        triton_poi_fused_argmin_0[grid(4096)](arg0_1, buf0, 4096, XBLOCK=
            128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
