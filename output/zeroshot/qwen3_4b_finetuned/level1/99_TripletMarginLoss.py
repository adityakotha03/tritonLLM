import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def triton_poi_fused_add_mul_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8192
    x1 = xindex // 8192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_sub_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tmp0 - tmp1
    tl.store(in_out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp5 - tmp4
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768, 8192), (8192, 1))
    assert_size_stride(arg1_1, (32768, 8192), (8192, 1))
    assert_size_stride(arg2_1, (32768, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        assert_size_stride(buf0, (32768, 8192), (8192, 1))
        buf1 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        assert_size_stride(buf1, (32768, 8192), (8192, 1))
        buf2 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        assert_size_stride(buf2, (32768, 8192), (8192, 1))
        buf3 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        del arg1_1
        buf4 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf5 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf6 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf7 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf8 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf9 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf10 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf11 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf12 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf13 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf14 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf15 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf16 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf17 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf18 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf19 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf20 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf21 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf22 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf23 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf24 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf25 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf26 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf27 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf28 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf29 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf30 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf31 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf32 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf33 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf34 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf35 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf36 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf37 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf38 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf39 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf40 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf41 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf42 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf43 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf44 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf45 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf46 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf47 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf48 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf49 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf50 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf51 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf52 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf53 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf54 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf55 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf56 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf57 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf58 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf59 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf60 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf61 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf62 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf63 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf64 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf65 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf66 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf67 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf68 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf69 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf70 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf71 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf72 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf73 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf74 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf75 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf76 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf77 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf78 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf79 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf80 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf81 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf82 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf83 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf84 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf85 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf86 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf87 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf88 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf89 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf90 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf91 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf92 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf93 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf94 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf95 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf96 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf97 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf98 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf99 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf100 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf101 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf102 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf103 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf104 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf105 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf106 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf107 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf108 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf109 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf110 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf111 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf112 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf113 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf114 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf115 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf116 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf117 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf118 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf119 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf120 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf121 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf122 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf123 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf124 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf125 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf126 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf127 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf128 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf129 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf130 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf131 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf132 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf133 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf134 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf135 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf136 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf137 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf138 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf139 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf140 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf141 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf142 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf143 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf144 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf145 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf146 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf147 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf148 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf149 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf150 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf151 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf152 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf153 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf154 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf155 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf156 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf157 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf158 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf159 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf160 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf161 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf162 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf163 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf164 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf165 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf166 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf167 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf168 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf169 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf170 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf171 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf172 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf173 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf174 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf175 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf176 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf177 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf178 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf179 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf180 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf181 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf182 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf183 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf184 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf185 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf186 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf187 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf188 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf189 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf190 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf191 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf192 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf193 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf194 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf195 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf196 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf197 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf198 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf199 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf200 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf201 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf202 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf203 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf204 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf205 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf206 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf207 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf208 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf209 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf210 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf211 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf212 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf213 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf214 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf215 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf216 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf217 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf218 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf219 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf220 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf221 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf222 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf223 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf224 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf225 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf226 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf227 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf228 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf229 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf230 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf231 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf232 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf233 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf234 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf235 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf236 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf237 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf238 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf239 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf240 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf241 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf242 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf243 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf244 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf245 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf246 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf247 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf248 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf249 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf250 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf251 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf252 = torch.ops.aten.mm.default(arg0_1, arg1_1)
        del arg0_1
        buf253 = torch.ops.aten.mm.default(arg0_1, arg2_1)
        del arg0_1
        buf254 = torch.ops.aten.mm.default(arg1_1, arg2_1)
        del arg1_1
        buf255 = torch.ops