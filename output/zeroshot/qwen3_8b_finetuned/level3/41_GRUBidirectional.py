import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_relu_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x3 = xindex
    x1 = xindex
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x1, None)
    tmp2 = tl.load(in_ptr1 + x3, None)
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + x2, tmp6, None)
    tl.store(out_ptr1 + x3, tmp6, None)


@triton.jit
def triton_poi_fused_add_mul_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tl.load(in_ptr1 + x1, None)
    tmp2 = tl.load(in_ptr2 + x3, None)
    tmp3 = tl.load(in_ptr3 + x0, None)
    tmp4 = tl.load(in_ptr4 + x1, None)
    tmp5 = tl.load(in_ptr5 + x3, None)
    tmp6 = tmp0 + tmp1
    tmp7 = tmp6 + tmp2
    tmp8 = tmp3 + tmp4
    tmp9 = tmp8 + tmp5
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp7)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp9)
    tl.store(out_ptr0 + x0, tmp11, None)
    tl.store(out_ptr1 + x3, tmp13, None)


@triton.jit
def triton_poi_fused_mul_add_relu_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x3 = xindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr1 + x1, None)
    tmp2 = tl.load(in_ptr2 + x3, None)
    tmp3 = tmp0 * tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + x2, tmp6, None)
    tl.store(out_ptr1 + x3, tmp6, None)


def triton_per_fused_add_relu_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x3 = xindex
    x1 = xindex
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x1, None)
    tmp2 = tl.load(in_ptr1 + x3, None)
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + x2, tmp6, None)
    tl.store(out_ptr1 + x3, tmp6, None)


def triton_per_fused_add_mul_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tl.load(in_ptr1 + x1, None)
    tmp2 = tl.load(in_ptr2 + x3, None)
    tmp3 = tl.load(in_ptr3 + x0, None)
    tmp4 = tl.load(in_ptr4 + x1, None)
    tmp5 = tl.load(in_ptr5 + x3, None)
    tmp6 = tmp0 + tmp1
    tmp7 = tmp6 + tmp2
    tmp8 = tmp3 + tmp4
    tmp9 = tmp8 + tmp5
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp7)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp9)
    tl.store(out_ptr0 + x0, tmp11, None)
    tl.store(out_ptr1 + x3, tmp13, None)


def triton_per_fused_mul_add_relu_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x3 = xindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr1 + x1, None)
    tmp2 = tl.load(in_ptr2 + x3, None)
    tmp3 = tmp0 * tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + x2, tmp6, None)
    tl.store(out_ptr1 + x3, tmp6, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
    args.clear()
    assert_size_stride(primals_1, (20, 128), (128, 1))
    assert_size_stride(primals_2, (20, 128), (128, 1))
    assert_size_stride(primals_3, (20, 256), (256, 1))
    assert_size_stride(primals_4, (20, 256), (256, 1))
    assert_size_stride(primals_5, (20, 256), (256, 1))
    assert_size_stride(primals_6, (20, 256), (256, 1))
    assert_size_stride(primals_7, (20, 128), (128, 1))
    assert_size_stride(primals_8, (20, 128), (128, 1))
    assert_size_stride(primals_9, (20, 256), (256, 1))
    assert_size_stride(primals_10, (20, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((20, 20, 256), (5120, 256, 1), torch.float32)
        buf1 = empty_strided_cuda((20, 20, 256), (5120, 256, 1), torch.float32)
        buf2 = empty_strided_cuda((20, 20, 256), (5120, 256, 1), torch.float32)
        buf3 = empty_strided_cuda((20, 20, 256), (5120, 256, 1), torch.float32)
        buf4 = empty_strided_cuda((20, 20, 256), (5120, 256, 1), torch.float32)
        del primals_1
        del primals_2
        get_rawbuf = buf0
        del buf0
        triton_poi_fused_add_relu_0[grid(20)](get_rawbuf, primals_3,
            primals_4, buf1, buf2, 20, 256, XBLOCK=128)
        del primals_3
        del primals_4
        triton_poi_fused_add_mul_relu_1[grid(20)](buf1, primals_5,
            primals_6, primals_7, primals_8, primals_9, buf3, buf4, 20, 256,
            XBLOCK=128)
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_9
        triton_poi_fused_mul_add_relu_2[grid(20)](buf4, buf2, primals_10,
            buf0, buf1, 20, 256, XBLOCK=128)
        del primals_10
        buf5 = empty_strided_cuda((20, 20, 256), (5120, 256, 1), torch.float32)
        buf6 = empty_strided_cuda((20, 20, 256), (5120, 256, 1), torch.float32)
        triton_per_fused_add_relu_3[grid(20)](buf0, primals_3, primals_4,
            buf5, buf6, 20, 256, XBLOCK=128)
        del primals_3
        del primals_4
        triton_per_fused_add_mul_relu_4[grid(20)](buf5, primals_5,
            primals_6, primals_7, primals_8, primals_9, buf3, buf4, 20, 256,
            XBLOCK=128)
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_9
        triton_per_fused_mul_add_relu_5[grid(20)](buf4, buf6, primals_10,
            buf0, buf1, 20, 256, XBLOCK=128)
        del primals_10
    return buf0, buf1, buf2, buf3, buf4, reinterpret_tensor(buf5, (20, 20,
        256), (5120, 256, 1), 0), reinterpret_tensor(buf6, (20, 20, 256),
        (5120, 256, 1), 0)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True,
        batch_first=False):
        super(ModelNew, self).__init__()
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.randn((hidden_size * 2,
            input_size * 2), dtype=torch.float32))
        self.weight_hh_l0 = nn.Parameter(torch.randn((hidden_size * 2,
            hidden_size * 2), dtype=torch.float32))
        self.weight_ih_l1 = nn.Parameter(torch.randn((hidden_size * 2,
            input_size * 2), dtype=torch.float32))
        self.weight_hh_l1 = nn.Parameter(torch.randn((hidden_size * 2,
            hidden_size * 2), dtype=torch.float32))
        self.weight_ih_l2 = nn.Parameter(torch.randn((hidden_size * 2,
            input_size * 2), dtype=torch.float32))
        self.weight_hh_l2 = nn.Parameter(torch.randn((hidden_size * 2,
            hidden_size * 2), dtype=torch.float32))

    def forward(self, input_0, input_1):
        primals_1 = self.weight_ih_l0
        primals_2 = self.weight_hh_l0
        primals_3 = self.weight_ih_l1
        primals_4 = self.weight_hh_l1
        primals_5 = self.weight_ih_l2
        primals_6 = self.weight_hh_l2
        primals_7 = primals_1
        primals_8 = primals_2
        primals_9 = primals_3
        primals_10 = primals_4
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, input_0, input_1])
        return output[0], output[5]