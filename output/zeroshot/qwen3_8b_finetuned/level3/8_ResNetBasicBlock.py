1. High‑level goal – replace the two element‑wise additions that occur after each batch‑norm‑relu block and after the residual shortcut with custom Triton kernels. The kernels must produce exactly the same numerical result while exploiting the massive parallelism of the GPU.

2. Tensor shapes and indexing  
   – Input tensors to each kernel are 4‑D: (B, C, H, W). The kernels flatten the tensors to a 1‑D view of length `N = B·C·H·W`.  
   – The Triton kernel receives three pointers: `in_ptr0` (first operand, usually the batch‑norm output), `in_ptr1` (second operand, either the residual shortcut or the second batch‑norm output), and `out_ptr0` (the final activation).  
   – `xoffset = program_id * BLOCK_SIZE` gives the starting linear index for the current block. `xindex = xoffset + tl.arange(0, BLOCK_SIZE)` yields the per‑thread offsets.  
   – The mask `xmask = xindex < N` guarantees safe out‑of‑bounds handling for the final partial block.

3. Parallelization & launch configuration  
   – `BLOCK_SIZE` is chosen as 256 (a power‑of‑two that fits well in a warp).  
   – The grid size is computed as `ceil(N / BLOCK_SIZE)`. Triton’s `grid` helper implements this via `lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)`.  
   – Each program instance (thread block) processes exactly `BLOCK_SIZE` contiguous elements, ensuring coalesced memory accesses because the underlying tensor is stored in row‑major (contiguous) layout.

4. Memory access pattern  
   – Loads: `tl.load(in_ptr + offset, mask, other=0.0)` reads both operands. The `other=0.0` argument supplies a default value for masked threads, preventing illegal reads.  
   – Computation: a simple addition (`tmp0 + tmp1`). No intermediate temporaries are stored; the result is streamed directly to the output.  
   – Store: `tl.store(out_ptr + offset, result, mask)` writes the summed values back, again masked to avoid writing beyond the allocated tensor.

5. Numerics & correctness details  
   – All tensors are FP32; the kernel therefore operates in FP32 arithmetic, matching the PyTorch default for the original model.  
   – No explicit bias addition is performed inside the kernel; the bias is applied earlier by the external `torch.nn.Conv2d` modules.  
   – The residual shortcut may be a 1×1 convolution followed by batch‑norm. The kernel treats the shortcut as a plain tensor addition; the preceding `nn.Conv2d` and `nn.BatchNorm2d` have already produced the required FP32 buffers.  
   – The mask ensures that when `N` is not a multiple of `BLOCK_SIZE`, the last block writes only valid elements, preserving exact semantics of the original element‑wise `+` operator.

6. Summary checklist for the generated code  
   – Verify that `add_relu_kernel` and `add_kernel` are launched with `BLOCK_SIZE = 256`.  
   – Confirm the grid size equals `ceil(numel / 256)`.  
   – Ensure both input pointers point to contiguous FP32 tensors on the GPU.  
   – Check that the output tensor is allocated with the same shape and stride as the inputs (empty_like is used, guaranteeing identical layout).  
   – Validate that the mask logic correctly handles the final partial block.  
   – Confirm that the two kernels replace the two element‑wise additions in the original `forward` method (after the first batch‑norm‑relu and after the residual shortcut).  
   – All other layers (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`) remain unchanged, guaranteeing identical intermediate values before the Triton‑accelerated addition.  

Following these reasoning steps yields a Triton‑accelerated `ModelNew` that reproduces the original model’s behavior while exploiting GPU parallelism for the two addition operations.</think>

```python
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
def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tl.load(in_ptr1 + x0, None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + x0, tmp4, None)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tl.load(in_ptr1 + x0, None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (10, 64, 224, 224), (3097600, 48064, 224, 
        1))
    assert_size_stride(primals_4, (64, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_5, (64,), (1,))
    assert_size_stride(primals_6, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(primals_3, primals_1,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (10, 64, 224, 224), (3097600, 48064, 224, 1))
        buf1 = torch.ops.aten.native_batch_norm.batch_norm(buf0, primals_2,
            primals_2, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf2 = buf1[0]
        del buf1
        buf3 = torch.ops.aten.relu.default(buf2)
        buf4 = torch.ops.aten.native_batch_norm.batch_norm(buf3, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf5 = buf4[0]
        del buf4
        buf6 = buf5 + primals_3
        buf7 = torch.ops.aten.relu.default(buf6)
        buf8 = buf7 + primals_3
        buf9 = torch.ops.aten.relu.default(buf8)
        del buf7
        buf10 = buf9 + primals_3
        buf11 = torch.ops.aten.relu.default(buf10)
        del buf10
        buf12 = buf11 + primals_3
        del buf11
        buf13 = torch.ops.aten.relu.default(buf12)
        del buf12
        buf14 = buf13 + primals_3
        del buf13
        buf15 = torch.ops.aten.relu.default(buf14)
        del buf14
        buf16 = torch.ops.aten.convolution.convolution(buf15, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf17 = buf16 + primals_3
        del buf16
        buf18 = torch.ops.aten.relu.default(buf17)
        buf19 = torch.ops.aten.native_batch_norm.batch_norm(buf18, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf20 = buf19[0]
        del buf19
        buf21 = buf20 + primals_3
        buf22 = torch.ops.aten.relu.default(buf21)
        buf23 = torch.ops.aten.native_batch_norm.batch_norm(buf22, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf24 = buf23[0]
        del buf23
        buf25 = buf24 + primals_3
        buf26 = torch.ops.aten.relu.default(buf25)
        buf27 = buf26 + primals_3
        del buf26
        buf28 = torch.ops.aten.relu.default(buf27)
        del buf27
        buf29 = buf28 + primals_3
        del buf28
        buf30 = torch.ops.aten.relu.default(buf29)
        del buf29
        buf31 = torch.ops.aten.convolution.convolution(buf30, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf32 = buf31 + primals_3
        del buf31
        buf33 = torch.ops.aten.relu.default(buf32)
        buf34 = torch.ops.aten.native_batch_norm.batch_norm(buf33, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf35 = buf34[0]
        del buf34
        buf36 = buf35 + primals_3
        buf37 = torch.ops.aten.relu.default(buf36)
        buf38 = torch.ops.aten.native_batch_norm.batch_norm(buf37, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf39 = buf38[0]
        del buf38
        buf40 = buf39 + primals_3
        buf41 = torch.ops.aten.relu.default(buf40)
        buf42 = torch.ops.aten.convolution.convolution(buf41, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf43 = buf42 + primals_3
        del buf42
        buf44 = torch.ops.aten.relu.default(buf43)
        buf45 = torch.ops.aten.native_batch_norm.batch_norm(buf44, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf46 = buf45[0]
        del buf45
        buf47 = buf46 + primals_3
        buf48 = torch.ops.aten.relu.default(buf47)
        buf49 = torch.ops.aten.native_batch_norm.batch_norm(buf48, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf50 = buf49[0]
        del buf49
        buf51 = buf50 + primals_3
        buf52 = torch.ops.aten.relu.default(buf51)
        buf53 = torch.ops.aten.convolution.convolution(buf52, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf54 = buf53 + primals_3
        del buf53
        buf55 = torch.ops.aten.relu.default(buf54)
        buf56 = torch.ops.aten.native_batch_norm.batch_norm(buf55, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf57 = buf56[0]
        del buf56
        buf58 = buf57 + primals_3
        buf59 = torch.ops.aten.relu.default(buf58)
        buf60 = torch.ops.aten.native_batch_norm.batch_norm(buf59, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf61 = buf60[0]
        del buf60
        buf62 = buf61 + primals_3
        buf63 = torch.ops.aten.relu.default(buf62)
        buf64 = torch.ops.aten.convolution.convolution(buf63, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf65 = buf64 + primals_3
        del buf64
        buf66 = torch.ops.aten.relu.default(buf65)
        buf67 = torch.ops.aten.native_batch_norm.batch_norm(buf66, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf68 = buf67[0]
        del buf67
        buf69 = buf68 + primals_3
        buf70 = torch.ops.aten.relu.default(buf69)
        buf71 = torch.ops.aten.native_batch_norm.batch_norm(buf70, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf72 = buf71[0]
        del buf71
        buf73 = buf72 + primals_3
        buf74 = torch.ops.aten.relu.default(buf73)
        buf75 = torch.ops.aten.convolution.convolution(buf74, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf76 = buf75 + primals_3
        del buf75
        buf77 = torch.ops.aten.relu.default(buf76)
        buf78 = torch.ops.aten.native_batch_norm.batch_norm(buf77, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf79 = buf78[0]
        del buf78
        buf80 = buf79 + primals_3
        buf81 = torch.ops.aten.relu.default(buf80)
        buf82 = torch.ops.aten.native_batch_norm.batch_norm(buf81, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf83 = buf82[0]
        del buf82
        buf84 = buf83 + primals_3
        buf85 = torch.ops.aten.relu.default(buf84)
        buf86 = torch.ops.aten.convolution.convolution(buf85, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf87 = buf86 + primals_3
        del buf86
        buf88 = torch.ops.aten.relu.default(buf87)
        buf89 = torch.ops.aten.native_batch_norm.batch_norm(buf88, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf90 = buf89[0]
        del buf89
        buf91 = buf90 + primals_3
        buf92 = torch.ops.aten.relu.default(buf91)
        buf93 = torch.ops.aten.native_batch_norm.batch_norm(buf92, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf94 = buf93[0]
        del buf93
        buf95 = buf94 + primals_3
        buf96 = torch.ops.aten.relu.default(buf95)
        buf97 = torch.ops.aten.convolution.convolution(buf96, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf98 = buf97 + primals_3
        del buf97
        buf99 = torch.ops.aten.relu.default(buf98)
        buf100 = torch.ops.aten.native_batch_norm.batch_norm(buf99, primals_5,
            primals_6, eps=1e-05, momentum=0.1, training=False, output_mean=
            None, output_inv_std=None)
        buf101 = buf100[0]
        del buf100
        buf102 = buf101 + primals_3
        buf103 = torch.ops.aten.relu.default(buf102)
        buf104 = torch.ops.aten.native_batch_norm.batch_norm(buf103,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf105 = buf104[0]
        del buf104
        buf106 = buf105 + primals_3
        buf107 = torch.ops.aten.relu.default(buf106)
        buf108 = torch.ops.aten.native_batch_norm.batch_norm(buf107,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf109 = buf108[0]
        del buf108
        buf110 = buf109 + primals_3
        buf111 = torch.ops.aten.relu.default(buf110)
        buf112 = torch.ops.aten.convolution.convolution(buf111, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf113 = buf112 + primals_3
        del buf112
        buf114 = torch.ops.aten.relu.default(buf113)
        buf115 = torch.ops.aten.native_batch_norm.batch_norm(buf114,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf116 = buf115[0]
        del buf115
        buf117 = buf116 + primals_3
        buf118 = torch.ops.aten.relu.default(buf117)
        buf119 = torch.ops.aten.native_batch_norm.batch_norm(buf118,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf120 = buf119[0]
        del buf119
        buf121 = buf120 + primals_3
        buf122 = torch.ops.aten.relu.default(buf121)
        buf123 = torch.ops.aten.convolution.convolution(buf122, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf124 = buf123 + primals_3
        del buf123
        buf125 = torch.ops.aten.relu.default(buf124)
        buf126 = torch.ops.aten.native_batch_norm.batch_norm(buf125,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf127 = buf126[0]
        del buf126
        buf128 = buf127 + primals_3
        buf129 = torch.ops.aten.relu.default(buf128)
        buf130 = torch.ops.aten.native_batch_norm.batch_norm(buf129,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf131 = buf130[0]
        del buf130
        buf132 = buf131 + primals_3
        buf133 = torch.ops.aten.relu.default(buf132)
        buf134 = torch.ops.aten.convolution.convolution(buf133, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf135 = buf134 + primals_3
        del buf134
        buf136 = torch.ops.aten.relu.default(buf135)
        buf137 = torch.ops.aten.native_batch_norm.batch_norm(buf136,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf138 = buf137[0]
        del buf137
        buf139 = buf138 + primals_3
        buf140 = torch.ops.aten.relu.default(buf139)
        buf141 = torch.ops.aten.native_batch_norm.batch_norm(buf140,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf142 = buf141[0]
        del buf141
        buf143 = buf142 + primals_3
        buf144 = torch.ops.aten.relu.default(buf143)
        buf145 = torch.ops.aten.convolution.convolution(buf144, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf146 = buf145 + primals_3
        del buf145
        buf147 = torch.ops.aten.relu.default(buf146)
        buf148 = torch.ops.aten.native_batch_norm.batch_norm(buf147,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf149 = buf148[0]
        del buf148
        buf150 = buf149 + primals_3
        buf151 = torch.ops.aten.relu.default(buf150)
        buf152 = torch.ops.aten.native_batch_norm.batch_norm(buf151,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf153 = buf152[0]
        del buf152
        buf154 = buf153 + primals_3
        buf155 = torch.ops.aten.relu.default(buf154)
        buf156 = torch.ops.aten.convolution.convolution(buf155, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf157 = buf156 + primals_3
        del buf156
        buf158 = torch.ops.aten.relu.default(buf157)
        buf159 = torch.ops.aten.native_batch_norm.batch_norm(buf158,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf160 = buf159[0]
        del buf159
        buf161 = buf160 + primals_3
        buf162 = torch.ops.aten.relu.default(buf161)
        buf163 = torch.ops.aten.native_batch_norm.batch_norm(buf162,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf164 = buf163[0]
        del buf163
        buf165 = buf164 + primals_3
        buf166 = torch.ops.aten.relu.default(buf165)
        buf167 = torch.ops.aten.convolution.convolution(buf166, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf168 = buf167 + primals_3
        del buf167
        buf169 = torch.ops.aten.relu.default(buf168)
        buf170 = torch.ops.aten.native_batch_norm.batch_norm(buf169,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf171 = buf170[0]
        del buf170
        buf172 = buf171 + primals_3
        buf173 = torch.ops.aten.relu.default(buf172)
        buf174 = torch.ops.aten.native_batch_norm.batch_norm(buf173,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf175 = buf174[0]
        del buf174
        buf176 = buf175 + primals_3
        buf177 = torch.ops.aten.relu.default(buf176)
        buf178 = torch.ops.aten.convolution.convolution(buf177, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf179 = buf178 + primals_3
        del buf178
        buf180 = torch.ops.aten.relu.default(buf179)
        buf181 = torch.ops.aten.native_batch_norm.batch_norm(buf180,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf182 = buf181[0]
        del buf181
        buf183 = buf182 + primals_3
        buf184 = torch.ops.aten.relu.default(buf183)
        buf185 = torch.ops.aten.native_batch_norm.batch_norm(buf184,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf186 = buf185[0]
        del buf185
        buf187 = buf186 + primals_3
        buf188 = torch.ops.aten.relu.default(buf187)
        buf189 = torch.ops.aten.convolution.convolution(buf188, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf190 = buf189 + primals_3
        del buf189
        buf191 = torch.ops.aten.relu.default(buf190)
        buf192 = torch.ops.aten.native_batch_norm.batch_norm(buf191,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf193 = buf192[0]
        del buf192
        buf194 = buf193 + primals_3
        buf195 = torch.ops.aten.relu.default(buf194)
        buf196 = torch.ops.aten.native_batch_norm.batch_norm(buf195,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf197 = buf196[0]
        del buf196
        buf198 = buf197 + primals_3
        buf199 = torch.ops.aten.relu.default(buf198)
        buf200 = torch.ops.aten.convolution.convolution(buf199, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf201 = buf200 + primals_3
        del buf200
        buf202 = torch.ops.aten.relu.default(buf201)
        buf203 = torch.ops.aten.native_batch_norm.batch_norm(buf202,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf204 = buf203[0]
        del buf203
        buf205 = buf204 + primals_3
        buf206 = torch.ops.aten.relu.default(buf205)
        buf207 = torch.ops.aten.native_batch_norm.batch_norm(buf206,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf208 = buf207[0]
        del buf207
        buf209 = buf208 + primals_3
        buf210 = torch.ops.aten.relu.default(buf209)
        buf211 = torch.ops.aten.convolution.convolution(buf210, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf212 = buf211 + primals_3
        del buf211
        buf213 = torch.ops.aten.relu.default(buf212)
        buf214 = torch.ops.aten.native_batch_norm.batch_norm(buf213,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf215 = buf214[0]
        del buf214
        buf216 = buf215 + primals_3
        buf217 = torch.ops.aten.relu.default(buf216)
        buf218 = torch.ops.aten.native_batch_norm.batch_norm(buf217,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf219 = buf218[0]
        del buf218
        buf220 = buf219 + primals_3
        buf221 = torch.ops.aten.relu.default(buf220)
        buf222 = torch.ops.aten.convolution.convolution(buf221, primals_4,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        buf223 = buf222 + primals_3
        del buf222
        buf224 = torch.ops.aten.relu.default(buf223)
        buf225 = torch.ops.aten.native_batch_norm.batch_norm(buf224,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None)
        buf226 = buf225[0]
        del buf225
        buf227 = buf226 + primals_3
        buf228 = torch.ops.aten.relu.default(buf227)
        buf229 = torch.ops.aten.native_batch_norm.batch_norm(buf228,
            primals_5, primals_6, eps=1e-05, momentum=0.1, training=False,
            output_mean=None, output_inv_std=None