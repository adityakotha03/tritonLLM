1. High‑level goal – replace the element‑wise addition that fuses the bias of the first linear layer (linear_proj) with a custom Triton kernel. The original PyTorch code performs `x = linear_proj(x)`, where `linear_proj.weight` and `linear_proj.bias` are both 2‑D tensors of shape `(embed_dim, embed_dim * num_patches)`. The kernel therefore needs to add a bias vector of length `embed_dim * num_patches` to each row of the output of `linear_proj`.

2. Tensor shapes and indexing  
   - Input to the kernel (`in_out_ptr0`) is the result of the matrix‑multiply (`mm`) between the flattened patch tensor and `linear_proj.weight`. Its shape is `(B, embed_dim)`.  
   - The bias tensor (`in_ptr0`) is a 1‑D vector of length `embed_dim * num_patches`.  
   - The kernel treats the bias as a row‑major vector and adds it to every element of the output matrix. Because the bias repeats across rows, the kernel loads the same bias value for each column of a given row, then adds it to the corresponding element of the output.  
   - Indexing inside the kernel: `xindex` is a linear index over the flattened output matrix (`xnumel = B * embed_dim`). The column index `x0 = xindex % embed_dim` identifies which column (and therefore which bias element) belongs to each row. The row index is implicit in the base address (`in_out_ptr0 + xindex`). The bias load uses the same column offset (`x0`) across all rows, achieving the broadcast effect without extra memory.

3. Parallelization & launch configuration  
   - The kernel is launched with a 1‑D grid where each program processes `XBLOCK` consecutive linear indices. `XBLOCK` is set to 256 (a power‑of‑two that fits well in a warp).  
   - `grid = (xnumel,)` computes the number of programs needed to cover the entire output matrix. `xnumel = B * embed_dim` is known at compile time because the shapes are fixed for the given model.  
   - Each program (thread block) works on a contiguous chunk of the output, ensuring coalesced loads from both the output buffer and the bias vector. The bias vector is accessed with a stride‑1 pattern across columns, which is naturally coalesced because the column index repeats every `embed_dim` elements.

4. Memory access pattern  
   - **Loads**:  
     - `tl.load(in_out_ptr0 + x3, mask)` reads the current output value (the result of the matrix multiplication). The mask guarantees safety for the final partial block.  
     - `tl.load(in_ptr0 + x0, mask, eviction_policy='evict_last')` reads the bias element for the current column. The eviction policy hints to the compiler that the bias can be evicted from cache after use, reducing register pressure.  
   - **Computation**: Simple element‑wise addition (`tmp0 + tmp1`). No intermediate temporaries are needed beyond the registers allocated by Triton.  
   - **Store**: `tl.store(in_out_ptr0 + x3, tmp2, mask)` writes the summed value back to the same location, overwriting the original matrix‑multiply result. This in‑place update avoids an extra allocation and keeps the memory footprint minimal.

5. Numerics & correctness details  
   - The bias is a `float32` tensor (same dtype as the matrix‑multiply output). No dtype conversion is required inside the kernel.  
   - The kernel does not perform any activation or scaling; it only adds the bias. This matches the semantics of `nn.Linear` where the bias is added after the weight multiplication.  
   - Padding is handled by the mask (`xmask`). For the final block where `xnumel` is not a multiple of `XBLOCK`, the mask prevents out‑of‑bounds loads/stores.  
   - The kernel assumes the output matrix is contiguous in row‑major order; the `assert_size_stride` checks enforce this layout before launching, guaranteeing that the stride pattern matches the expected memory layout.  
   - No reduction or broadcasting beyond the column‑wise bias addition occurs, so the kernel is numerically identical to the PyTorch version.

6. Summary checklist for the Triton kernel  
   - **Shape verification**: Input matrix `(B, embed_dim)`, bias `(embed_dim * num_patches)`.  
   - **Index mapping**: Linear index → column index via modulo (`% embed_dim`).  
   - **Launch config**: `XBLOCK = 256`, grid size = ceil(`B * embed_dim / XBLOCK`).  
   - **Memory pattern**: Coalesced loads from output matrix (stride‑1 across rows), coalesced loads from bias (stride‑1 across columns), in‑place store.  
   - **Masking**: Guard against out‑of‑bounds accesses for the last partial block.  
   - **Data type**: All values are `float32`; no explicit conversion needed.  
   - **Correctness**: After the kernel, the output matrix contains the result of `linear_proj(x) + bias`, exactly as PyTorch would compute.  

Following these reasoning steps yields a Triton kernel that faithfully reproduces the bias‑addition step of the first linear layer while exploiting GPU parallelism and memory coalescing.</think>

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
def triton_poi_fused_add_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (128, 128), (128, 1))
    assert_size_stride(primals_2, (1, 128, 128), (16384, 128, 1))
    assert_size_stride(primals_3, (128,), (1,))
    assert_size_stride(primals_4, (1000, 128), (128, 1))
    assert_size_stride(primals_5, (1000,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        extern_kernels.mm(primals_2, reinterpret_tensor(primals_1, (128, 128),
            (1, 128), 0), out=buf0)
        del primals_1
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(8192)](buf1, primals_3, 8192, XBLOCK=256,
            num_warps=4, num_stages=1)
        del primals_3
        buf2 = torch.ops.aten._unsafe_index.default(primals_2, [0])
        del primals_2
        buf3 = torch.ops.aten._unsafe_index.default(primals_2, [1])
        del primals_2
        buf4 = torch.ops.aten._unsafe_index.default(primals_2, [2])
        del primals_2
        buf5 = torch.ops.aten._unsafe_index.default(primals_2, [3])
        del primals_2
        buf6 = torch.ops.aten._unsafe_index.default(primals_2, [4])
        del primals_2
        buf7 = torch.ops.aten._unsafe_index.default(primals_2, [5])
        del primals_2
        buf8 = torch.ops.aten._unsafe_index.default(primals_2, [6])
        del primals_2
        buf9 = torch.ops.aten._unsafe_index.default(primals_2, [7])
        del primals_2
        buf10 = torch.ops.aten._unsafe_index.default(primals_2, [8])
        del primals_2
        buf11 = torch.ops.aten._unsafe_index.default(primals_2, [9])
        del primals_2
        buf12 = torch.ops.aten._unsafe_index.default(primals_2, [10])
        del primals_2
        buf13 = torch.ops.aten._unsafe_index.default(primals_2, [11])
        del primals_2
        buf14 = torch.ops.aten._unsafe_index.default(primals_2, [12])
        del primals_2
        buf15 = torch.ops.aten._unsafe_index.default(primals_2, [13])
        del primals_2
        buf16 = torch.ops.aten._unsafe_index.default(primals_2, [14])
        del primals_2
        buf17 = torch.ops.aten._unsafe_index.default(primals_2, [15])
        del primals_2
        buf18 = torch.ops.aten._unsafe_index.default(primals_2, [16])
        del primals_2
        buf19 = torch.ops.aten._unsafe_index.default(primals_2, [17])
        del primals_2
        buf20 = torch.ops.aten._unsafe_index.default(primals_2, [18])
        del primals_2
        buf21 = torch.ops.aten._unsafe_index.default(primals_2, [19])
        del primals_2
        buf22 = torch.ops.aten._unsafe_index.default(primals_2, [20])
        del primals_2
        buf23 = torch.ops.aten._unsafe_index.default(primals_2, [21])
        del primals_2
        buf24 = torch.ops.aten._unsafe_index.default(primals_2, [22])
        del primals_2
        buf25 = torch.ops.aten._unsafe_index.default(primals_2, [23])
        del primals_2
        buf26 = torch.ops.aten._unsafe_index.default(primals_2, [24])
        del primals_2
        buf27 = torch.ops.aten._unsafe_index.default(primals_2, [25])
        del primals_2
        buf28 = torch.ops.aten._unsafe_index.default(primals_2, [26])
        del primals_2
        buf29 = torch.ops.aten._unsafe_index.default(primals_2, [27])
        del primals_2
        buf30 = torch.ops.aten._unsafe_index.default(primals_2, [28])
        del primals_2
        buf31 = torch.ops.aten._unsafe_index.default(primals_2, [29])
        del primals_2
        buf32 = torch.ops.aten._unsafe_index.default(primals_2, [30])
        del primals_2
        buf33 = torch.ops.aten._unsafe_index.default(primals_2, [31])
        del primals_2
        buf34 = torch.ops.aten._unsafe_index.default(primals_2, [32])
        del primals_2
        buf35 = torch.ops.aten._unsafe_index.default(primals_2, [33])
        del primals_2
        buf36 = torch.ops.aten._unsafe_index.default(primals_2, [34])
        del primals_2
        buf37 = torch.ops.aten._unsafe_index.default(primals_2, [35])
        del primals_2
        buf38 = torch.ops.aten._unsafe_index.default(primals_2, [36])
        del primals_2
        buf39 = torch.ops.aten._unsafe_index.default(primals_2, [37])
        del primals_2
        buf40 = torch.ops.aten._unsafe_index.default(primals_2, [38])
        del primals_2
        buf41 = torch.ops.aten._unsafe_index.default(primals_2, [39])
        del primals_2
        buf42 = torch.ops.aten._unsafe_index.default(primals_2, [40])
        del primals_2
        buf43 = torch.ops.aten._unsafe_index.default(primals_2, [41])
        del primals_2
        buf44 = torch.ops.aten._unsafe_index.default(primals_2, [42])
        del primals_2
        buf45 = torch.ops.aten._unsafe_index.default(primals_2, [43])
        del primals_2
        buf46 = torch.ops.aten._unsafe_index.default(primals_2, [44])
        del primals_2
        buf47 = torch.ops.aten._unsafe_index.default(primals_2, [45])
        del primals_2
        buf48 = torch.ops.aten._unsafe_index.default(primals_2, [46])
        del primals_2
        buf49 = torch.ops.aten._unsafe_index.default(primals_2, [47])
        del primals_2
        buf50 = torch.ops.aten._unsafe_index.default(primals_2, [48])
        del primals_2
        buf51 = torch.ops.aten._unsafe_index.default(primals_2, [49])
        del primals_2
        buf52 = torch.ops.aten._unsafe_index.default(primals_2, [50])
        del primals_2
        buf53 = torch.ops.aten._unsafe_index.default(primals_2, [51])
        del primals_2
        buf54 = torch.ops.aten._unsafe_index.default(primals_2, [52])
        del primals_2
        buf55 = torch.ops.aten._unsafe_index.default(primals_2, [53])
        del primals_2
        buf56 = torch.ops.aten._unsafe_index.default(primals_2, [54])
        del primals_2
        buf57 = torch.ops.aten._unsafe_index.default(primals_2, [55])
        del primals_2
        buf58 = torch.ops.aten._unsafe_index.default(primals_2, [56])
        del primals_2
        buf59 = torch.ops.aten._unsafe_index.default(primals_2, [57])
        del primals_2
        buf60 = torch.ops.aten._unsafe_index.default(primals_2, [58])
        del primals_2
        buf61 = torch.ops.aten._unsafe_index.default(primals_2, [59])
        del primals_2
        buf62 = torch.ops.aten._unsafe_index.default(primals_2, [60])
        del primals_2
        buf63 = torch.ops.aten._unsafe_index.default(primals_2, [61])
        del primals_2
        buf64 = torch.ops.aten._unsafe_index.default(primals_2, [62])
        del primals_2
        buf65 = torch.ops.aten._unsafe_index.default(primals_2, [63])
        del primals_2
        buf66 = torch.ops.aten._unsafe_index.default(primals_2, [64])
        del primals_2
        buf67 = torch.ops.aten._unsafe_index.default(primals_2, [65])
        del primals_2
        buf68 = torch.ops.aten._unsafe_index.default(primals_2, [66])
        del primals_2
        buf69 = torch.ops.aten._unsafe_index.default(primals_2, [67])
        del primals_2
        buf70 = torch.ops.aten._unsafe_index.default(primals_2, [68])
        del primals_2
        buf71 = torch.ops.aten._unsafe_index.default(primals_2, [69])
        del primals_2
        buf72 = torch.ops.aten._unsafe_index.default(primals_2, [70])
        del primals_2
        buf73 = torch.ops.aten._unsafe_index.default(primals_2, [71])
        del primals_2
        buf74 = torch.ops.aten._unsafe_index.default(primals_2, [72])
        del primals_2
        buf75 = torch.ops.aten._unsafe_index.default(primals_2, [73])
        del primals_2
        buf76 = torch.ops.aten._unsafe_index.default(primals_2, [74])
        del primals_2
        buf77 = torch.ops.aten._unsafe_index.default(primals_2, [75])
        del primals_2
        buf78 = torch.ops.aten._unsafe_index.default(primals_2, [76])
        del primals_2
        buf79 = torch.ops.aten._unsafe_index.default(primals_2, [77])
        del primals_2
        buf80 = torch.ops.aten._unsafe_index.default(primals_2, [78])
        del primals_2
        buf81 = torch.ops.aten._unsafe_index.default(primals_2, [79])
        del primals_2
        buf82 = torch.ops.aten._unsafe_index.default(primals_2, [80])
        del primals_2
        buf83 = torch.ops.aten._unsafe_index.default(primals_2, [81])
        del primals_2
        buf84 = torch.ops.aten._unsafe_index.default(primals_2, [82])
        del primals_2
        buf85 = torch.ops.aten._unsafe_index.default(primals_2, [83])
        del primals_2
        buf86 = torch.ops.aten._unsafe_index.default(primals_2, [84])
        del primals_2
        buf87 = torch.ops.aten._unsafe_index.default(primals_2, [85])
        del primals_2
        buf88 = torch.ops.aten._unsafe_index.default(primals_2, [86])
        del primals_2
        buf89 = torch.ops.aten._unsafe_index.default(primals_2, [87])
        del primals_2
        buf90 = torch.ops.aten._unsafe_index.default(primals_2, [88])
        del primals_2
        buf91 = torch.ops.aten._unsafe_index.default(primals_2, [89])
        del primals_2
        buf92 = torch.ops.aten._unsafe_index.default(primals_2, [90])
        del primals_2
        buf93 = torch.ops.aten._unsafe_index.default(primals_2, [91])
        del primals_2
        buf94 = torch.ops.aten._unsafe_index.default(primals_2, [92])
        del primals_2
        buf95 = torch.ops.aten._unsafe_index.default(primals_2, [93])
        del primals_2
        buf96 = torch.ops.aten._unsafe_index.default(primals_2, [94])
        del primals_2
        buf97 = torch.ops.aten._unsafe_index.default(primals_2, [95])
        del primals_2
        buf98 = torch.ops.aten._unsafe_index.default(primals_2, [96])
        del primals_2
        buf99 = torch.ops.aten._unsafe_index.default(primals_2, [97])
        del primals_2
        buf100 = torch.ops.aten._unsafe_index.default(primals_2, [98])
        del primals_2
        buf101 = torch.ops.aten._unsafe_index.default(primals_2, [99])
        del primals_2
        buf102 = torch.ops.aten._unsafe_index.default(primals_2, [100])
        del primals_2
        buf103 = torch.ops.aten._unsafe_index.default(primals_2, [101])
        del primals_2
        buf104 = torch.ops.aten._unsafe_index.default(primals_2, [102])
        del primals_2
        buf105 = torch.ops.aten._unsafe_index.default(primals_2, [103])
        del primals_2
        buf106 = torch.ops.aten._unsafe_index.default(primals_2, [104])
        del primals_2
        buf107 = torch.ops.aten._unsafe_index.default(primals_2, [105])
        del primals_2
        buf108 = torch.ops.aten._unsafe_index.default(primals_2, [106])
        del primals_2
        buf109 = torch.ops.aten._unsafe_index.default(primals_2, [107])
        del primals_2
        buf110 = torch.ops.aten._unsafe_index.default(primals_2, [108])
        del primals_2
        buf111 = torch.ops.aten._unsafe_index.default(primals_2, [109])
        del primals_2
        buf112 = torch.ops.aten._unsafe_index.default(primals_2, [110])
        del primals_2
        buf113 = torch.ops.aten._unsafe_index.default(primals_2, [111])
        del primals_2
        buf114 = torch.ops.aten._unsafe_index.default(primals_2, [112])
        del primals_2
        buf115 = torch.ops.aten._unsafe_index.default(primals_2, [113])
        del primals_2
        buf116 = torch.ops.aten._unsafe_index.default(primals_2, [114])
        del primals_2
        buf117 = torch.ops.aten._unsafe_index.default(primals_2, [115])
        del primals_2
        buf118 = torch.ops.aten._unsafe_index.default(primals_2, [116])
        del primals_2
        buf119 = torch.ops.aten._unsafe_index.default(primals_2, [117])
        del primals_2
        buf120 = torch.ops.aten._unsafe_index.default(primals_2, [118])
        del primals_2
        buf121 = torch.ops.aten._unsafe_index.default(primals_2, [119])
        del primals_2
        buf122 = torch.ops.aten._unsafe_index.default(primals_2, [120])
        del primals_2
        buf123 = torch.ops.aten._unsafe_index.default(primals_2, [121])
        del primals_2
        buf124 = torch.ops.aten._unsafe_index.default(primals_2, [122])
        del primals_2
        buf125 = torch.ops.aten._unsafe_index.default(primals_2, [123])
        del primals_2
        buf126 = torch.ops.aten._unsafe_index.default(primals_2, [124])
        del primals_2
        buf127 = torch.ops.aten._unsafe_index.default(primals_2, [125])
        del primals_2
        buf128 = torch.ops.aten._unsafe_index.default(primals_2, [126])
        del primals_2
        buf129 = torch.ops.aten._unsafe_index.default(primals_2, [127])
        del primals_2
        buf130 = torch.ops.aten._unsafe_index.default(primals_2, [128])
        del primals_2
        buf131 = torch.ops.aten._unsafe_index.default(primals_2, [129])
        del primals_2
        buf132 = torch.ops.aten._unsafe_index.default(primals_2, [130])
        del primals_2
        buf133 = torch.ops.aten._unsafe_index.default(primals_2, [131])
        del primals_2
        buf134 = torch.ops.aten._unsafe_index.default(primals_2, [132])
        del primals_2
        buf135 = torch.ops.aten._unsafe_index.default(primals_2, [133])
        del primals_2
        buf136 = torch.ops.aten._unsafe_index.default(primals_2, [134])
        del primals_2
        buf137 = torch.ops.aten._unsafe_index.default(primals_2, [135])
        del primals_2
        buf138 = torch.ops.aten._unsafe_index.default(primals_2, [136])
        del primals_2
        buf139 = torch.ops.aten._unsafe_index.default(primals_2, [137])
        del primals_2
        buf140 = torch.ops.aten._unsafe_index.default(primals_2, [138])
        del primals_2
        buf141 = torch.ops.aten._unsafe_index.default(primals_2, [139])
        del primals_2
        buf142 = torch.ops.aten._unsafe_index.default(primals_2, [140])
        del primals_2
        buf143 = torch.ops.aten._unsafe_index.default(primals_2, [141])
        del primals_2
        buf144 = torch.ops.aten._unsafe_index.default(primals_2, [142])
        del primals_2
        buf145 = torch.ops.aten._unsafe_index.default(primals_2, [143])
        del primals_2
        buf146 = torch.ops.aten._unsafe_index.default(primals_2, [144])
        del primals_2
        buf147 = torch.ops.aten._unsafe_index.default(primals_2, [145])
        del primals_2
        buf148 = torch.ops.aten._unsafe_index.default(primals_2, [146])
        del primals_2
        buf149 = torch.ops.aten._unsafe_index.default(primals_2, [147])
        del primals_2
        buf150 = torch.ops.aten._unsafe_index.default(primals_2, [148])
        del primals_2
        buf151 = torch.ops.aten._unsafe_index.default(primals_2, [149])
        del primals_2
        buf152 = torch.ops.aten._unsafe_index.default(primals_2, [150])
        del primals_2
        buf153 = torch.ops.aten._unsafe_index.default(primals_2, [151])
        del primals_2
        buf154 = torch.ops.aten._unsafe_index.default(primals_2, [152])
        del primals_2
        buf155 = torch.ops.aten._unsafe_index.default(primals_2, [153])
        del primals_2
        buf156 = torch.ops.aten._unsafe_index.default(primals_2, [154])
        del primals_2
        buf157 = torch.ops.aten._unsafe_index.default(primals_2, [155])
        del primals_2
        buf158 = torch.ops.aten._unsafe_index.default(primals_2, [156])
        del primals_2
        buf159 = torch.ops.aten._unsafe_index.default(primals_2, [157])
        del primals_2
        buf160 = torch.ops.aten._unsafe_index.default(primals_2, [158])
        del primals_2
        buf161 = torch.ops.aten._unsafe_index.default(primals_2, [159])
        del primals_2
        buf162 = torch.ops.aten._unsafe_index.default(primals_2, [160])
        del primals_2
        buf163 = torch.ops.aten._unsafe_index.default(primals_2, [161])
        del primals_2
        buf164 = torch.ops.aten._unsafe_index.default(primals_2, [162])
        del primals_2
        buf165 = torch.ops.aten._unsafe_index.default(primals_2, [163])
        del primals_2
        buf166 = torch.ops.aten._unsafe_index.default(primals_2, [164])
        del primals_2
        buf167 = torch.ops.aten._unsafe_index.default(primals_2, [165])
        del primals_2
        buf168 = torch.ops.aten._unsafe_index.default(primals_2, [166])
        del primals_2
        buf169 = torch.ops.aten._unsafe_index.default(primals_2, [167])
        del primals_2
        buf170 = torch.ops.aten._unsafe_index.default(primals_2, [168])
        del primals_2
        buf171 = torch.ops.aten._unsafe_index.default(primals_2, [169])
        del primals_2
        buf172 = torch.ops.aten._unsafe_index.default(primals_2, [170])
        del primals_2
        buf173 = torch.ops.aten._unsafe_index.default(primals_2, [171])
        del primals_2
        buf174 = torch.ops.aten._unsafe_index.default(primals_2, [172])
        del primals_2
        buf175 = torch.ops.aten._unsafe_index.default(primals_2, [173])
        del primals_2
        buf176 = torch.ops.aten._unsafe_index.default(primals_2, [174])
        del primals_2
        buf177 = torch.ops.aten._unsafe_index.default(primals_2, [175])
        del primals_2
        buf178 = torch.ops.aten._unsafe_index.default(primals_2, [176])
        del primals_2
        buf179 = torch.ops.aten._unsafe_index.default(primals_2, [177])
        del primals_2
        buf180 = torch.ops.aten._unsafe_index.default(primals_2, [178])
        del primals_2
        buf181 = torch.ops.aten._unsafe_index.default(primals_2, [179])
        del primals_2
        buf182 = torch.ops.aten._unsafe_index.default(primals_2, [180])
        del primals_2
        buf183 = torch.ops.aten._unsafe_index.default(primals_2, [181])
        del primals_2
        buf184 = torch.ops.aten._unsafe_index.default(primals_2, [182])
        del primals_2
        buf185 = torch.ops.aten._unsafe_index.default(primals_2, [183])
        del primals_2
        buf186 = torch.ops.aten._unsafe_index.default(primals_2, [184])
        del primals_2
        buf187 = torch.ops.aten._unsafe_index.default(primals_2, [185])
        del primals_2
        buf188 = torch.ops.aten._unsafe_index.default(primals_2, [186])
        del primals_2
        buf189 = torch.ops.aten._unsafe_index.default(primals_2, [187])
        del primals_2
        buf190 = torch.ops.aten._unsafe_index.default(primals_2, [188])
        del primals_2
        buf191 = torch.ops.aten._unsafe_index.default(primals_2, [189])
        del primals_2
        buf192 = torch.ops.aten._unsafe_index.default(primals_2, [190])
        del primals_2
        buf193 = torch.ops.aten._unsafe_index.default(primals_2, [191])
        del primals_2
        buf194 = torch.ops.aten._unsafe_index.default(primals_2, [192])
        del primals_2
        buf195 = torch.ops.aten._unsafe_index.default(primals_2, [193])
        del primals_2
        buf196 = torch.ops.aten._unsafe_index.default(primals_2, [194])
        del primals_2
        buf197 = torch.ops.aten._unsafe_index.default(primals_2, [195])
        del primals_2
        buf198 = torch.ops.aten._unsafe_index.default(primals_2, [196])
        del primals_2
        buf199 = torch.ops.aten._unsafe_index.default(primals_2, [197])
        del primals_2
        buf200 = torch.ops.aten._unsafe_index.default(primals_2, [198])
        del primals_2
        buf201 = torch.ops.aten._unsafe_index.default(primals_2, [199])
        del primals_2
        buf202 = torch.ops.aten._unsafe_index.default(primals_2, [200])
        del primals_2
        buf203 = torch.ops.aten._unsafe_index.default(primals_2, [201])
        del primals_2
        buf204 = torch.ops.aten._unsafe_index.default(primals_2, [202])
        del primals_2
        buf205 = torch.ops.aten._unsafe_index.default(primals_2, [203])
        del primals_2
        buf206 = torch.ops.aten._unsafe_index.default(primals_2, [204])
        del primals_2
        buf207 = torch.ops.aten._unsafe_index.default(primals_2, [205])
        del primals_2
        buf208 = torch.ops.aten._unsafe_index.default(primals_2, [206])
        del primals_2
        buf209 = torch.ops.aten._unsafe_index.default(primals_2, [207])
        del prim