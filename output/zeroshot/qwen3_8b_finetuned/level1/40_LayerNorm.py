import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_add_div_mean_mul_sub_var_0(in_out_ptr0, in_ptr0,
    in_ptr1, out_ptr1, out_ptr2, xnumel, ynumel, XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr):
    xnumel = 2048
    ynumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex
    tmp0 = tl.load(in_out_ptr0 + (x2 + 262144 * y0), xmask & ymask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + y0, ymask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, YBLOCK])
    tmp4 = tl.where(xmask & ymask, tmp3, 0)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, YBLOCK])
    tmp7 = tl.where(xmask & ymask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 262144, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp0 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, YBLOCK])
    tmp16 = tl.where(xmask & ymask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = 64.0
    tmp19 = tmp17 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1, 1], 64, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tl.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tl.store(in_out_ptr0 + (x2 + 262144 * y0), tmp26, xmask & ymask)
    tl.store(out_ptr1 + y0, tmp1, ymask)
    tl.store(out_ptr2 + y0, tmp12, ymask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 64, 256, 256), (4194304, 65536, 256,
        1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        buf1 = empty_strided_cuda((64,), (1,), torch.float32)
        buf2 = empty_strided_cuda((64,), (1,), torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_sub_var_0[grid(2048, 64)](buf0,
            primals_2, primals_3, buf1, buf2, 2048, 64, XBLOCK=128,
            YBLOCK=64, num_warps=4, num_stages=1)
        del primals_2
        del primals_3
    return buf0, primals_1, buf1, buf2


class ModelNew(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, input_0):
        primals_2 = self.ln.weight
        primals_3 = self.ln.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]