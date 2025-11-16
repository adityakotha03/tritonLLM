import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024 % 256
    x3 = xindex // 262144
    x1 = xindex // 256
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024 * x1), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + 256 * x0), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x0 + 1024 * x1), tmp4 & xmask, other=0.0)
    tmp9 = tmp8 + tmp7
    tl.store(out_ptr0 + (x0 + 256 * x3), tmp9, xmask)


@triton.jit
def triton_poi_fused_tanh_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = xindex // 262144
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.7615941559557649
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 - tmp8
    tmp10 = 0.0
    tmp11 = tl.where(xmask, tmp9, tmp10)
    tmp12 = triton_helpers.maximum(tmp11, tmp8)
    tmp13 = -tmp11
    tmp14 = triton_helpers.maximum(tmp13, tmp8)
    tmp15 = tl_math.log1p(tmp12)
    tmp16 = tl_math.log1p(tmp14)
    tmp17 = tmp15 - tmp16
    tmp18 = tmp0 - tmp1
    tmp19 = tmp18 * tmp3
    tmp20 = tmp19 * tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = tl_math.rsqrt(tmp22)
    tmp24 = tmp17 * tmp23
    tl.store(in_out_ptr0 + x2, tmp24, xmask)


@triton.jit
def triton_poi_fused_tanh_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = xindex // 262144
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.7615941559557649
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 - tmp8
    tmp10 = 0.0
    tmp11 = tl.where(xmask, tmp9, tmp10)
    tmp12 = triton_helpers.maximum(tmp11, tmp8)
    tmp13 = -tmp11
    tmp14 = triton_helpers.maximum(tmp13, tmp8)
    tmp15 = tl_math.log1p(tmp12)
    tmp16 = tl_math.log1p(tmp14)
    tmp17 = tmp15 - tmp16
    tmp18 = tmp0 - tmp1
    tmp19 = tmp18 * tmp3
    tmp20 = tmp19 * tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = tl_math.rsqrt(tmp22)
    tmp24 = tmp17 * tmp23
    tl.store(in_out_ptr0 + x2, tmp24, xmask)


@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = xindex // 256
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256 * x1), tmp4 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 % 262144 + 1024 * (x2 // 262144)), tmp4 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp11 = tmp9 + tmp10
    tl.store(out_ptr0 + x2, tmp11, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8) = args
    args.clear()
    assert_size_stride(primals_1, (256, 1296), (1296, 1))
    assert_size_stride(primals_2, (1296,), (1,))
    assert_size_stride(primals_3, (256, 8, 1024), (8192, 1024, 1))
    assert_size_stride(primals_4, (256,), (1,))
    assert_size_stride(primals_5, (128, 256), (256, 1))
    assert_size_stride(primals_6, (128,), (1,))
    assert_size_stride(primals_7, (8, 256), (256, 1))
    assert_size_stride(primals_8, (256,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 1296), (1296, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (256, 8192), (8192,
            1), 0), reinterpret_tensor(primals_1, (8192, 1296), (1, 8192), 0
            ), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        extern_kernels.addmm(primals_6, reinterpret_tensor(buf0, (8, 1296),
            (1296, 1), 0), reinterpret_tensor(primals_5, (1296, 128), (1, 
            1296), 0), alpha=1, beta=1, out=buf1)
        del primals_6
        buf2 = empty_strided_cuda((8, 1296), (1296, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (8, 1024), (1024, 1),
            0), reinterpret_tensor(primals_2, (1024, 1296), (1, 1024), 0),
            out=buf2)
        del primals_2
        buf3 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        extern_kernels.addmm(primals_4, buf2, reinterpret_tensor(primals_5,
            (1296, 128), (1, 1296), 0), alpha=1, beta=1, out=buf3)
        del primals_4
        buf4 = empty_strided_cuda((256, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (256, 8192), (8192, 
            1), 0), reinterpret_tensor(primals_2, (8192, 1024), (1, 8192), 
            0), out=buf4)
        buf5 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        extern_kernels.addmm(primals_7, reinterpret_tensor(buf4, (8, 1024),
            (1024, 1), 0), reinterpret_tensor(primals_5, (1024, 128), (1, 
            1024), 0), alpha=1, beta=1, out=buf5)
        del primals_7
        buf6 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (256, 8192), (8192, 
            1), 0), reinterpret_tensor(primals_2, (8192, 256), (1, 8192), 0),
            out=buf6)
        buf7 = buf6
        del buf6
        get_raw_stream(0)
        triton_poi_fused_tanh_1[grid(2097152)](buf7, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf8 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (256, 8192), (8192, 
            1), 0), reinterpret_tensor(primals_2, (8192, 256), (1, 8192), 0),
            out=buf8)
        buf9 = buf8
        del buf8
        triton_poi_fused_tanh_1[grid(2097152)](buf9, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf10 = empty_strided_cuda((256, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (256, 8192), (8192, 
            1), 0), reinterpret_tensor(primals_2, (8192, 1024), (1, 8192), 0),
            out=buf10)
        buf11 = buf10
        del buf10
        triton_poi_fused_tanh_2[grid(2097152)](buf11, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf12 = reinterpret_tensor(buf4, (8192, 256), (256, 1), 0)
        del buf4
        extern_kernels.mm(reinterpret_tensor(buf7, (256, 256), (256, 1), 0),
            reinterpret_tensor(primals_2, (256, 1296), (1, 256), 0), out=buf12
            )
        buf13 = buf12
        del buf12
        triton_poi_fused_tanh_2[grid(2097152)](buf13, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf14 = reinterpret_tensor(buf9, (8192, 256), (256, 1), 0)
        del buf9
        extern_kernels.mm(reinterpret_tensor(buf7, (256, 256), (256, 1), 0),
            reinterpret_tensor(primals_2, (256, 1296), (1, 256), 0), out=buf14
            )
        buf15 = buf14
        del buf14
        triton_poi_fused_tanh_2[grid(2097152)](buf15, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf16 = reinterpret_tensor(buf11, (8192, 256), (256, 1), 0)
        del buf11
        extern_kernels.mm(reinterpret_tensor(buf7, (256, 256), (256, 1), 0),
            reinterpret_tensor(primals_2, (256, 1296), (1, 256), 0), out=buf16
            )
        buf17 = buf16
        del buf16
        triton_poi_fused_tanh_2[grid(2097152)](buf17, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf18 = empty_strided_cuda((8, 256), (256, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf13, (8, 1024), (1024, 1), 0),
            reinterpret_tensor(primals_5, (1024, 128), (1, 1024), 0), out=buf18
            )
        buf19 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        extern_kernels.addmm(primals_4, buf15, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf19)
        del primals_4
        buf20 = buf15
        del buf15
        triton_poi_fused_tanh_2[grid(2097152)](buf20, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf21 = reinterpret_tensor(buf17, (8192, 256), (256, 1), 0)
        del buf17
        extern_kernels.mm(reinterpret_tensor(buf7, (256, 256), (256, 1), 0),
            reinterpret_tensor(primals_2, (256, 1296), (1, 256), 0), out=buf21
            )
        buf22 = buf21
        del buf21
        triton_poi_fused_tanh_2[grid(2097152)](buf22, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf23 = reinterpret_tensor(buf18, (8192, 128), (128, 1), 0)
        del buf18
        extern_kernels.addmm(primals_4, buf20, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf23)
        buf24 = reinterpret_tensor(buf23, (8, 128), (128, 1), 0)
        del buf23
        extern_kernels.addmm(primals_4, buf22, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf24)
        buf25 = reinterpret_tensor(buf24, (8, 128), (128, 1), 0)
        del buf24
        extern_kernels.addmm(primals_4, buf19, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf25)
        buf26 = reinterpret_tensor(buf25, (8, 128), (128, 1), 0)
        del buf25
        extern_kernels.addmm(primals_4, buf13, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf26)
        buf27 = reinterpret_tensor(buf26, (8, 128), (128, 1), 0)
        del buf26
        extern_kernels.addmm(primals_4, buf10, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf27)
        buf28 = reinterpret_tensor(buf27, (8, 128), (128, 1), 0)
        del buf27
        extern_kernels.addmm(primals_4, buf6, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf28)
        buf29 = reinterpret_tensor(buf28, (8, 128), (128, 1), 0)
        del buf28
        extern_kernels.addmm(primals_4, buf3, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf29)
        buf30 = empty_strided_cuda((256, 1024), (1024, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (256, 8192), (8192, 
            1), 0), reinterpret_tensor(primals_2, (8192, 1024), (1, 8192), 0),
            out=buf30)
        buf31 = empty_strided_cuda((256, 1296), (1296, 1), torch.float32)
        triton_poi_fused_cat_0[grid(32768)](buf29, buf30, primals_5, buf31,
            32768, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_5
        buf32 = reinterpret_tensor(buf31, (8192, 256), (256, 1), 0)
        del buf31
        extern_kernels.mm(reinterpret_tensor(buf30, (8192, 256), (256, 1), 0
            ), reinterpret_tensor(primals_2, (256, 1296), (1, 256), 0), out=
            buf32)
        buf33 = buf32
        del buf32
        triton_poi_fused_tanh_2[grid(2097152)](buf33, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf34 = reinterpret_tensor(buf29, (8192, 128), (128, 1), 0)
        del buf29
        extern_kernels.addmm(primals_4, buf33, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf34)
        buf35 = reinterpret_tensor(buf34, (8, 128), (128, 1), 0)
        del buf34
        extern_kernels.addmm(primals_4, buf2, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf35)
        buf36 = empty_strided_cuda((8, 256), (256, 1), torch.float32)
        extern_kernels.mm(buf35, reinterpret_tensor(primals_5, (128, 1024),
            (1, 128), 0), out=buf36)
        buf37 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        extern_kernels.addmm(primals_7, buf36, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf37)
        buf38 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        extern_kernels.addmm(primals_7, buf3, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf38)
        buf39 = empty_strided_cuda((256, 1296), (1296, 1), torch.float32)
        triton_poi_fused_cat_3[grid(262144)](primals_5, buf1, buf3, buf36,
            buf39, 262144, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_5
        buf40 = reinterpret_tensor(buf38, (8192, 256), (256, 1), 0)
        del buf38
        extern_kernels.mm(reinterpret_tensor(buf37, (8192, 256), (256, 1), 0
            ), reinterpret_tensor(primals_2, (256, 1296), (1, 256), 0), out=
            buf40)
        buf41 = buf40
        del buf40
        triton_poi_fused_tanh_2[grid(2097152)](buf41, primals_8, 2097152,
            XBLOCK=2048, num_warps=4, num_stages=1)
        del primals_8
        buf42 = reinterpret_tensor(buf37, (8192, 128), (128, 1), 0)
        del buf37
        extern_kernels.addmm(primals_4, buf41, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf42)
        buf43 = reinterpret_tensor(buf42, (8, 128), (128, 1), 0)
        del buf42
        extern_kernels.addmm(primals_7, buf1, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf43)
        buf44 = empty_strided_cuda((8, 256), (256, 1), torch.float32)
        extern_kernels.mm(buf43, reinterpret_tensor(primals_5, (128, 1024),
            (1, 128), 0), out=buf44)
        buf45 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        extern_kernels.addmm(primals_7, buf44, reinterpret_tensor(primals_5,
            (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf45)
    return (primals_3, primals_7, buf1, buf2, buf3, reinterpret_tensor(
        buf4, (8192, 256), (256, 1), 0), buf5, reinterpret_tensor(buf7, (8192,
        256), (256, 1), 0), reinterpret_tensor(buf9, (8192, 256), (256, 1), 
        0), buf10, reinterpret_tensor(buf11, (8192, 256), (256, 1), 0),
        buf12, reinterpret_tensor(buf13, (8192, 256), (256, 1), 0), buf14,
        reinterpret_tensor(buf15, (8192, 256), (256, 1), 0), buf16,
        reinterpret_tensor(buf17, (8192, 256), (256, 1), 0), buf18,
        reinterpret_tensor(buf20, (8192, 256), (256, 1), 0), buf21,
        reinterpret_tensor(buf22, (8192, 256), (256, 1), 0), buf23,
        reinterpret_tensor(buf24, (8192, 256), (256, 1), 0), buf25,
        reinterpret_tensor(buf26, (8192, 256), (256, 1), 0), buf27,
        reinterpret_tensor(buf28, (8192, 256), (256, 1), 0), buf29,
        buf30, buf31, reinterpret_tensor(buf32, (8192, 128), (128, 1), 0),
        buf33, reinterpret_tensor(buf34, (8192, 128), (128, 1), 0), buf35,
        buf36, buf37, reinterpret_tensor(buf40, (8192, 128), (128, 1), 0),
        buf41, reinterpret_tensor(buf42, (8192, 128), (128, 1), 0), buf43,
        buf44, buf45, primals_8)


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.

        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state

    def forward(self, input_0, input_1):
        primals_1 = self.i2h.weight
        primals_2 = self.i2h.bias
        primals_5 = self.h2o.weight
        primals_6 = self.h2o.bias
        primals_7 = input_1
        primals_8 = self.tanh.weight
        primals_4 = self.tanh.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8])
        return output[0]
