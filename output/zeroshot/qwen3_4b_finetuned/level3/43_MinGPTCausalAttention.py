import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024 * x2 + 786432 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 768 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024 * x2 + 8192 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 8 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024 * x2 + 8192 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 8 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024 * x2 + 786432 * y1), xmask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 768 * y3), tmp0, xmask)


@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp3 = tl.load(in_ptr0 + 4096 * x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1024 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (2048 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (3072 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = tmp14 * tmp1
    tmp16 = tl_math.exp(tmp15)
    tl.store(out_ptr0 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused__softmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4096 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1024 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (2048 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3072 + 4096 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = yindex // 768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 768 * x2 + 393216 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 512 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024 * x2 + 786432 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 768 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_add_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (768, 768), (768, 1))
    assert_size_stride(primals_2, (768,), (1,))
    assert_size_stride(primals_3, (128, 512, 768), (393216, 768, 1))
    assert_size_stride(primals_4, (768, 768), (768, 1))
    assert_size_stride(primals_5, (768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 512, 768), (393216, 768, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused_0[grid(1024, 768)](primals_3, buf0, 1024, 768,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del primals_3
        buf1 = empty_strided_cuda((768, 8, 768), (6144, 768, 1), torch.float32)
        triton_poi_fused_1[grid(1024, 8)](primals_1, buf1, 1024, 8, XBLOCK=
            16, YBLOCK=64, num_warps=4, num_stages=1)
        del primals_1
        buf2 = empty_strided_cuda((768, 8, 768), (6144, 768, 1), torch.float32)
        triton_poi_fused_2[grid(1024, 8)](primals_4, buf2, 1024, 8, XBLOCK=
            16, YBLOCK=64, num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((128, 512, 768), (393216, 768, 1), torch.
            float32)
        triton_poi_fused_3[grid(1024, 768)](buf0, buf3, 1024, 768, XBLOCK=
            32, YBLOCK=32, num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((128, 8, 512, 768), (3145728, 393216, 768,
            1), torch.float32)
        extern_kernels.bmm(reinterpret_tensor(buf3, (128, 8, 512), (393216,
            768, 1), 0), reinterpret_tensor(buf2, (128, 8, 768), (6144, 1, 
            768), 0), out=buf4)
        buf5 = empty_strided_cuda((128, 512, 1024), (524288, 1024, 1), torch
            .float32)
        triton_poi_fused__softmax_4[grid(4194304)](buf4, buf5, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf6 = reinterpret_tensor(buf4, (128, 512, 1024), (524288, 1024, 1), 0)
        del buf4
        triton_poi_fused__softmax_5[grid(4194304)](buf5, buf6, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf7 = buf5
        del buf5
        triton_poi_fused__softmax_5[grid(4194304)](buf6, buf7, 4194304,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf6
        buf8 = empty_strided_cuda((128, 512, 768), (393216, 768, 1), torch.
            float32)
        triton_poi_fused_clone_6[grid(1024, 512)](buf7, buf8, 1024, 512,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        buf9 = reinterpret_tensor(buf7, (128, 512, 768), (393216, 768, 1), 0)
        del buf7
        triton_poi_fused_clone_7[grid(512, 768)](buf1, buf9, 512, 768,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        buf10 = reinterpret_tensor(buf0, (128, 512, 768), (393216, 768, 1), 0)
        del buf0
        extern_kernels.bmm(reinterpret_tensor(buf8, (128, 512, 768), (393216
            , 768, 1), 0), reinterpret_tensor(buf1, (128, 768, 8), (6144, 1,
            768), 0), out=buf10)
        buf11 = reinterpret_tensor(buf1, (128, 768, 512), (393216, 512, 1), 0)
        del buf1
        triton_poi_fused_add_8[grid(98304)](buf11, primals_5, 98304, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_5
    return buf11, reinterpret_tensor(buf3, (128, 8, 512), (393216, 768, 1), 0
        ), buf9, reinterpret_tensor(buf8, (128, 768, 512), (393216, 1, 768), 0
        ), reinterpret_tensor(buf10, (128, 768, 512), (393216, 1, 768), 0
        ), reinterpret_tensor(buf2, (128, 768, 8), (6144, 1, 768), 0
        ), reinterpret_tensor(buf9, (128, 8, 768), (6144, 1, 768), 0
        ), reinterpret_tensor(buf2, (128, 8, 768), (6144, 768, 1), 0
        ), reinterpret_tensor(buf10, (128, 512, 768), (393216, 1, 768), 0)


class ModelNew(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, input_0):
        primals_1 = self.c_attn.weight
        primals_2 = self.c_attn.bias
        primals_4 = self.c_proj.weight
        primals_5 = self.c_proj.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
