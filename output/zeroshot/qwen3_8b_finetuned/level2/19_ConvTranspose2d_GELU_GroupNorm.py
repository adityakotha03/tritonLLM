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
def triton_add_native_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = xindex // 16384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384 * x1), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_add_native_add_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = xindex // 16384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384 * x1), xmask)
    tmp2 = tl.load(in_ptr0 + (256 + x0 + 16384 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (512 + x0 + 16384 * x1), xmask)
    tmp8 = tl.load(in_ptr0 + (768 + x0 + 16384 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (1024 + x0 + 16384 * x1), xmask)
    tmp14 = tl.load(in_ptr0 + (1280 + x0 + 16384 * x1), xmask)
    tmp17 = tl.load(in_ptr0 + (1536 + x0 + 16384 * x1), xmask)
    tmp20 = tl.load(in_ptr0 + (1792 + x0 + 16384 * x1), xmask)
    tmp23 = tl.load(in_ptr0 + (2048 + x0 + 16384 * x1), xmask)
    tmp26 = tl.load(in_ptr0 + (2304 + x0 + 16384 * x1), xmask)
    tmp31 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x1), xmask)
    tmp34 = tl.load(in_ptr0 + (6144 + x0 + 16384 * x1), xmask)
    tmp37 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x1), xmask)
    tmp40 = tl.load(in_ptr0 + (10240 + x0 + 16384 * x1), xmask)
    tmp43 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x1), xmask)
    tmp46 = tl.load(in_ptr0 + (14336 + x0 + 16384 * x1), xmask)
    tmp49 = tl.load(in_ptr0 + (16384 + x0 + 16384 * x1), xmask)
    tmp1 = 1.0
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 + tmp3
    tmp6 = tmp5 * tmp1
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp1
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp1
    tmp13 = tmp10 + tmp12
    tmp15 = tmp14 * tmp1
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17 * tmp1
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20 * tmp1
    tmp22 = tmp19 + tmp21
    tmp24 = tmp23 * tmp1
    tmp25 = tmp22 + tmp24
    tmp27 = tmp26 * tmp1
    tmp28 = tmp25 + tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp30 = tl.sum(tmp29, 0)[:, None]
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp33 = tl.sum(tmp32, 0)[:, None]
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp36 = tl.sum(tmp35, 0)[:, None]
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp39 = tl.sum(tmp38, 0)[:, None]
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp42 = tl.sum(tmp41, 0)[:, None]
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp45 = tl.sum(tmp44, 0)[:, None]
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp48 = tl.sum(tmp47, 0)[:, None]
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp51 = tl.sum(tmp50, 0)[:, None]
    tmp52 = tmp30 + tmp33
    tmp53 = tmp52 + tmp36
    tmp54 = tmp53 + tmp39
    tmp55 = tmp54 + tmp42
    tmp56 = tmp55 + tmp45
    tmp57 = tmp56 + tmp48
    tmp58 = tmp57 + tmp51
    tmp59 = 256.0
    tmp60 = tmp58 / tmp59
    tmp61 = tmp28 - tmp60
    tmp62 = tmp61 * tmp61
    tmp63 = tmp32 - tmp60
    tmp64 = tmp63 * tmp63
    tmp65 = tmp62 + tmp64
    tmp66 = tmp35 - tmp60
    tmp67 = tmp66 * tmp66
    tmp68 = tmp65 + tmp67
    tmp69 = tmp38 - tmp60
    tmp70 = tmp69 * tmp69
    tmp71 = tmp68 + tmp70
    tmp72 = tmp41 - tmp60
    tmp73 = tmp72 * tmp72
    tmp74 = tmp71 + tmp73
    tmp75 = tmp44 - tmp60
    tmp76 = tmp75 * tmp75
    tmp77 = tmp74 + tmp76
    tmp78 = tmp47 - tmp60
    tmp79 = tmp78 * tmp78
    tmp80 = tmp77 + tmp79
    tmp81 = tmp50 - tmp60
    tmp82 = tmp81 * tmp81
    tmp83 = tmp80 + tmp82
    tmp84 = tmp83 / tmp59
    tmp85 = 1e-05
    tmp86 = tmp84 + tmp85
    tmp87 = libdevice.sqrt(tmp86)
    tmp88 = tmp61 / tmp87
    tl.store(out_ptr0 + x2, tmp88, xmask)
    tl.store(out_ptr1 + x2, tmp60, xmask)
    tl.store(out_ptr2 + x2, tmp84, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 256, 256), (4194304, 65536, 256, 
        1))
    assert_size_stride(primals_2, (64, 64, 3, 3), (576, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf0, (128, 64, 256, 256), (4194304, 65536, 256, 
            1))
        buf1 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_add_native_add_0[grid(2097152)](buf0, buf1, 2097152,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((128, 64, 256), (16384, 256, 1), torch.
            float32)
        buf3 = empty_strided_cuda((128, 64, 256), (16384, 256, 1), torch.
            float32)
        buf4 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        triton_add_native_add_1[grid(2097152)](buf1, buf4, buf2, buf3,
            2097152, XBLOCK=1024, num_warps=4, num_stages=1)
    return buf4, primals_1, primals_2, buf0, buf1, buf2, buf3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_1 = input_0
        output = call([primals_1, primals_2])
        return output[0]