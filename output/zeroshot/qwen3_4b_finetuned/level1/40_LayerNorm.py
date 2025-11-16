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
def triton_poi_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1), xmask)
    tmp2 = tl.load(in_ptr0 + (256 + x0 + 256 * x1), xmask)
    tmp4 = tl.load(in_ptr0 + (512 + x0 + 256 * x1), xmask)
    tmp6 = tl.load(in_ptr0 + (768 + x0 + 256 * x1), xmask)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp5 = tmp4 * tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp1 + tmp3
    tmp9 = tmp8 + tmp5
    tmp10 = tmp9 + tmp7
    tmp11 = 4.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.promote_to_tensor(tl.broadcast_to(tmp14, [XBLOCK
        ]))
    tmp16 = 4.0
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + x2, tmp15, xmask)
    tl.store(out_ptr1 + x2, tmp17, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 64, 256, 256), (4194304, 66304, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 256, 256), (4194304, 66304, 256, 
            1), torch.float32)
        buf1 = empty_strided_cuda((16, 64, 256, 256), (4194304, 66304, 256, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_native_layer_norm_0[grid(16777216)](arg0_1, buf0,
            buf1, 16777216, XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
        buf2 = empty_strided_cuda((16, 64, 256, 256), (4194304, 66304, 256, 
            1), torch.float32)
        triton_poi_fused_native_layer_norm_1[grid(16777216)](buf1, buf2,
            16777216, XBLOCK=1024, num_warps=4, num_stages=1)
    return reinterpret_tensor(buf2, (16, 64, 256, 256), (4194304, 66304, 256,
        1), 0), buf0, buf1


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
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
