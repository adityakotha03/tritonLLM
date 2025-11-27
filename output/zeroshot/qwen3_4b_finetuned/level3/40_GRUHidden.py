import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 153600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_add_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 153600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 128 * x1), xmask)
    tmp2 = tl.load(in_ptr0 + (512 + x0 + 128 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (768 + x0 + 128 * x1), xmask)
    tmp4 = tmp0 + tmp1
    tmp5 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 256 * x1), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, 10, 128), (1280, 128, 1))
    assert_size_stride(arg1_1, (6, 10, 256), (2560, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 6, 256), (1536, 256, 1), torch.float32)
        get_ptr0 = reinterpret_tensor(buf0, (10, 6, 256), (1536, 256, 1), 0)
        triton_poi_fused_0[grid] = lambda meta: (153600,)
        triton_poi_fused_0[grid](arg1_1, get_ptr0, 153600, XBLOCK=256, num_warps=4,
            num_stages=1)
        buf1 = empty_strided_cuda((10, 6, 256), (1536, 256, 1), torch.float32)
        triton_poi_fused_1[grid] = lambda meta: (61440,)
        triton_poi_fused_1[grid](arg0_1, buf1, 61440, XBLOCK=256, num_warps=4,
            num_stages=1)
        del arg0_1
        del arg1_1
        buf2 = empty_strided_cuda((512, 10, 128), (1280, 128, 1), torch.float32)
        triton_poi_fused_add_2[grid] = lambda meta: (153600,)
        triton_poi_fused_add_2[grid](buf1, buf2, 153600, XBLOCK=256, num_warps=4,
            num_stages=1)
        buf3 = empty_strided_cuda((512, 10, 256), (2560, 256, 1), torch.float32)
        triton_poi_fused_add_3[grid] = lambda meta: (61440,)
        triton_poi_fused_add_3[grid](buf1, buf2, buf3, 61440, XBLOCK=256,
            num_warps=4, num_stages=1)
    return buf3, reinterpret_tensor(buf0, (10, 6, 256), (1536, 256, 1), 0)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(ModelNew, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
