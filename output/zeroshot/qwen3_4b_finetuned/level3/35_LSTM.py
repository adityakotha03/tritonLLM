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
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3840
    x1 = xindex // 3840
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6
    x3 = xindex // 192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + 2304), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x3, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2 % 384), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + x3, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 + tmp10
    tmp12 = tmp9 + tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + x2, tmp14, xmask)


@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr):
    ynumel = 10
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 10 * x1), xmask & ymask, eviction_policy
        ='evict_last')
    tl.store(out_ptr0 + (x1 + 576 * y0), tmp0, xmask & ymask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7
    = args
    args.clear()
    assert_size_stride(primals_1, (10, 128), (128, 1))
    assert_size_stride(primals_2, (256, 128), (128, 1))
    assert_size_stride(primals_3, (256, 128), (128, 1))
    assert_size_stride(primals_4, (6, 256), (256, 1))
    assert_size_stride(primals_5, (6, 256), (256, 1))
    assert_size_stride(primals_6, (10,), (1,))
    assert_size_stride(primals_7, (10, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 128), (128, 1), torch.float32)
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (128, 256
            ), (1, 128), 0), out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((6, 10, 256), (2560, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(3840)](buf0, primals_3, buf1, 3840,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_3
        buf2 = empty_strided_cuda((10, 256), (256, 1), torch.float32)
        triton_poi_fused_clone_1[grid(576)](primals_4, buf2, 576, XBLOCK=256,
            num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((6, 256), (256, 1), torch.float32)
        extern_kernels.addmm(primals_5, reinterpret_tensor(primals_5, (128,
            256), (1, 128), 0), reinterpret_tensor(primals_3, (256, 128), (
            1, 256), 0), alpha=1, beta=1, out=buf3)
        del primals_5
        buf4 = reinterpret_tensor(buf1, (512, 6, 256), (1536, 256, 1), 0)
        del buf1
        triton_poi_fused__to_copy_2[grid(3840)](buf3, buf4, 3840, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf3
        buf5 = empty_strided_cuda((512, 6, 256), (1536, 256, 1), torch.float32)
        triton_poi_fused__to_copy_3[grid(3840)](primals_3, buf5, 3840,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf6 = empty_strided_cuda((10, 6, 256), (1536, 256, 1), torch.float32)
        triton_poi_fused__unsafe_index_4[grid(576, 3840)](primals_6, buf5,
            buf6, primals_6, primals_7, primals_1, primals_1, buf6, 576, 3840,
            XBLOCK=128, YBLOCK=32, num_warps=4, num_stages=1)
        buf7 = empty_strided_cuda((10, 10), (10, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf6, (10, 576), (576, 1), 0),
            reinterpret_tensor(primals_7, (576, 10), (1, 576), 0), out=buf7)
        buf8 = reinterpret_tensor(buf6, (512, 10, 6), (60, 6, 1), 0)
        del buf6
        triton_poi_fused_5[grid(10, 576)](buf7, buf8, 10, 576, XBLOCK=256,
            YBLOCK=32, num_warps=4, num_stages=1)
        del buf7
    return reinterpret_tensor(buf8, (10, 10), (10, 1), 0), primals_1, buf4, buf5,
    primals_6, primals_7, buf2, primals_1


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_0):
        primals_1 = self.lstm.weight_ih_l0
        primals_2 = self.lstm.weight_hh_l0
        primals_3 = self.lstm.bias_hh_l0
        primals_4 = self.lstm.weight_ih_l1
        primals_5 = self.lstm.weight_hh_l1
        primals_6 = self.lstm.bias_hh_l1
        primals_7 = self.fc.weight
        primals_8 = self.fc.bias
        primals_9 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9])
        return output[0]
