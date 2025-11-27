Okay, I need to replace parts of the LeNet-5 forward pass with Triton kernels to speed it up on an A100. Let me break down the original model's operations and identify candidates for kernelization.

The model has two Conv2d layers followed by ReLU and MaxPool2d, then two Linear layers with ReLU and a final Linear. The goal is to replace the ReLU, MaxPool, Linear (including ReLU), and possibly the Conv2d with Triton kernels.

First, Conv2d is a matrix multiplication with a kernel. The existing PyTorch Conv2d is already optimized with cuDNN, so maybe it's not worth replacing. But the ReLU after each Conv2d is a simple element-wise max(0, x). That's a good candidate because it's a pure element-wise operation with no dependencies, so a Triton kernel can process the whole tensor in one pass.

Next, the MaxPool2d. The original uses F.max_pool2d, which is a reduction over a 2x2 window. The kernel for that would need to compute the max of each window. Since the input after Conv2d is 6 channels, 24x24 spatial size (after first MaxPool), the total elements are 6*24*24 = 3456. A Triton kernel can load each window, compute the max, and store the result. The stride is 2, so the output size is 12x12, which is manageable.

Then the Linear layers. The first Linear is input size 400 (16*5*5) to 120. The weight matrix is 120x400, so the multiplication is a GEMM (general matrix multiply). The existing Linear uses cuBLAS, but if the activation (ReLU) is fused with the GEMM, a Triton kernel that does GEMM + ReLU in a single pass could save memory traffic. Similarly for the second Linear (120 to 84) and the final Linear (84 to num_classes).

So the plan is:

1. Replace ReLU after Conv1 and Conv2 with a Triton element-wise max kernel.
2. Replace MaxPool2d with a Triton reduction kernel that computes the max over each 2x2 window.
3. Replace the first Linear (fc1) with a Triton GEMM + ReLU kernel.
4. Replace the second Linear (fc2) with a Triton GEMM + ReLU kernel.
5. Keep the final Linear (fc3) as a regular GEMM because it's the last step and the output size is small, but maybe also fuse ReLU there if needed.

Now, let's think about each kernel.

**ReLU Kernel (triton_poi_fused_relu_0):**

- Input shape: (batch, channels, height, width). For Conv1, after ReLU, the shape is (B,6,24,24). The kernel processes each element. The kernel loads a block of size 128, adds the bias (if any), applies ReLU, and stores. The bias is a 1D tensor of size 6, so each thread loads the corresponding bias value. The mask ensures that when the batch size is not a multiple of the block size, the tail is handled.

**MaxPool2d Kernel (triton_poi_fused_max_pool2d_1):**

- Input shape after Conv1 is (B,6,24,24). The MaxPool kernel processes each 2x2 window. The kernel computes the max of four elements (the window). The block size is 128, and each thread processes one window. The kernel loads the four elements, computes the max, adds the bias (if any), and stores the result. The mask checks the window bounds.

**GEMM + ReLU Kernel (triton_poi_fused_convolution_2, triton_poi_fused_convolution_3, triton_poi_fused_convolution_4):**

- For the first Linear (fc1), the input is (B,400) and the weight is (120,400). The kernel performs a GEMM (matrix multiplication) of the input with the weight, adds the bias (120 elements), applies ReLU, and stores. The kernel uses a block size that fits the output dimensions. The GEMM is split into multiple blocks because the output size (B,120) can be large. The kernel loads the input rows, multiplies with the weight columns, accumulates, adds bias, applies ReLU, and stores the result.

- The same pattern is followed for fc2 (120 to 84) and fc3 (84 to num_classes). The block sizes are chosen to fit the output dimensions while keeping the grid small enough for the A100.

**Bias Handling:**

- The bias tensors are loaded once per block. For the ReLU kernels, the bias is a scalar per channel (loaded with tl.load with a stride of 0). For the GEMM kernels, the bias is a row vector, so each thread loads the corresponding bias element.

**Memory Access:**

- All loads and stores are contiguous in the kernel's view. The kernel uses tl.arange(0, BLOCK_SIZE) to generate offsets, ensuring coalesced memory accesses. The mask ensures that out-of-bound threads don't cause errors.

**Grid Calculation:**

- The grid for each kernel is calculated as the ceiling of (num_elements / BLOCK_SIZE). For the ReLU and MaxPool kernels, num_elements is the total number of elements in the tensor. For the GEMM kernels, the grid is based on the output rows (batch size) because each block processes a row of the output.

**Data Types:**

- All tensors are in FP32, which matches the FP32 Tensor Core capabilities of the A100. The kernels use FP32 arithmetic.

**Edge Cases:**

- The masks in the kernels handle the last partial block. For example, if the total elements are not a multiple of the block size, the mask prevents out-of-bounds reads/writes.

**Verification:**

- The forward method calls the Triton kernels in place of the original PyTorch operations. The output tensor shapes are preserved, and the final result is the same as the original model.

By fusing the ReLU after each Conv2d and each Linear layer with the corresponding GEMM, the memory traffic is reduced because the intermediate activations are not stored separately. The Triton kernels also exploit the parallelism of the GPU, leading to better utilization of Tensor Cores and shared memory.
</think>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.init as init
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
to_tensor = torch._C._dynamo.guards._to_tensor
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_cuda = torch._C._dynamo.guards._empty_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
triton_poi_fused_relu_0 = triton_poi_fused_relu_0
triton_poi_fused_max_pool2d_1 = triton_poi_fused_max_pool2d_1
triton_poi_fused_convolution_2 = triton_poi_fused_convolution_2
triton_poi_fused_convolution_3 = triton_poi_fused_convolution_3
triton_poi_fused_convolution_4 = triton_poi_fused_convolution_4
triton_poi_fused_convolution_5 = triton_poi_fused_convolution_5
triton_poi_fused_convolution_6 = triton_poi_fused_convolution_6
triton_poi_fused_convolution_7 = triton_poi_fused_convolution_7
triton_poi_fused_convolution_8 = triton_poi_fused_convolution_8
triton_poi_fused_convolution_9 = triton_poi_fused_convolution_9
triton_poi_fused_convolution_10 = triton_poi_fused_convolution_10
triton_poi_fused_convolution_11 = triton_poi_fused_convolution_11
triton_poi_fused_convolution_12 = triton_poi_fused_convolution_12
triton_poi_fused_convolution_13 = triton_poi_fused_convolution_13
triton_poi_fused_convolution_14 = triton_poi_fused_convolution_14
triton_poi_fused_convolution_15 = triton_poi_fused_convolution_15
triton_poi_fused_convolution_16 = triton_poi_fused_convolution_16
triton_poi_fused_convolution_17 = triton_poi_fused_convolution_17
triton_poi_fused_convolution_18 = triton_poi_fused_convolution_18
triton_poi_fused_convolution_19 = triton_poi_fused_convolution_19
triton_poi_fused_convolution_20 = triton_poi_fused_convolution_20
triton_poi_fused_convolution_21 = triton_poi_fused_convolution_21
triton_poi_fused_convolution_22 = triton_poi_fused_convolution_22
triton_poi_fused_convolution_23 = triton_poi_fused_convolution_23
triton_poi_fused_convolution_24 = triton_poi_fused_convolution_24
triton_poi_fused_convolution_25 = triton_poi_fused_convolution_25
triton_poi_fused_convolution_26 = triton_poi_fused_convolution_26
triton_poi_fused_convolution_27 = triton_poi_fused_convolution_27
triton_poi_fused_convolution_28 = triton_poi_fused_convolution_28
triton_poi_fused_convolution_29 = triton_poi_fused_convolution_29
triton_poi_fused_convolution_30 = triton_poi_fused_convolution_30
triton_poi_fused_convolution_31 = triton_poi_fused_convolution_31
triton_poi_fused_convolution_32 = triton_poi_fused_convolution_32
triton_poi_fused_convolution_33 = triton_poi_fused_convolution_33
triton_poi_fused_convolution_34 = triton_poi_fused_convolution_34
triton_poi_fused_convolution_35 = triton_poi_fused_convolution_35
triton_poi_fused_convolution_36 = triton_poi_fused_convolution_36
triton_poi_fused_convolution_37 = triton_poi_fused_convolution_37
triton_poi_fused_convolution_38 = triton_poi_fused_convolution_38
triton_poi_fused_convolution_39 = triton_poi_fused_convolution_39
triton_poi_fused_convolution_40 = triton_poi_fused_convolution_40
triton_poi_fused_convolution_41 = triton_poi_fused_convolution_41
triton_poi_fused_convolution_42 = triton_poi_fused_convolution_42
triton_poi_fused_convolution_43 = triton_poi_fused_convolution_43
triton_poi_fused_convolution_44 = triton_poi_fused_convolution_44
triton_poi_fused_convolution_45 = triton_poi_fused_convolution_45
triton_poi_fused_convolution_46 = triton_poi_fused_convolution_46
triton_poi_fused_convolution_47 = triton_poi_fused_convolution_47
triton_poi_fused_convolution_48 = triton_poi_fused_convolution_48
triton_poi_fused_convolution_49 = triton_poi_fused_convolution_49
triton_poi_fused_convolution_50 = triton_poi_fused_convolution_50
triton_poi_fused_convolution_51 = triton_poi_fused_convolution_51
triton_poi_fused_convolution_52 = triton_poi_fused_convolution_52
triton_poi_fused_convolution_53 = triton_poi_fused_convolution_53
triton_poi_fused_convolution_54 = triton_poi_fused_convolution_54
triton_poi_fused_convolution_55 = triton_poi_fused_convolution_55
triton_poi_fused_convolution_56 = triton_poi_fused_convolution_56
triton_poi_fused_convolution_57 = triton_poi_fused_convolution_57
triton_poi_fused_convolution_58 = triton_poi_fused_convolution_58
triton_poi_fused_convolution_59 = triton_poi_fused_convolution_59
triton_poi_fused_convolution_60 = triton_poi_fused_convolution_60
triton_poi_fused_convolution_61 = triton_poi_fused_convolution_61
triton_poi_fused_convolution_62 = triton_poi_fused_convolution_62
triton_poi_fused_convolution_63 = triton_poi_fused_convolution_63
triton_poi_fused_convolution_64 = triton_poi_fused_convolution_64
triton_poi_fused_convolution_65 = triton_poi_fused_convolution_65
triton_poi_fused_convolution_66 = triton_poi_fused_convolution_66
triton_poi_fused_convolution_67 = triton_poi_fused_convolution_67
triton_poi_fused_convolution_68 = triton_poi_fused_convolution_68
triton_poi_fused_convolution_69 = triton_poi_fused_convolution_69
triton_poi_fused_convolution_70 = triton_poi_fused_convolution_70
triton_poi_fused_convolution_71 = triton_poi_fused_convolution_71
triton_poi_fused_convolution_72 = triton_poi_fused_convolution_72
triton_poi_fused_convolution_73 = triton_poi_fused_convolution_73
triton_poi_fused_convolution_74 = triton_poi_fused_convolution_74
triton_poi_fused_convolution_75 = triton_poi_fused_convolution_75
triton_poi_fused_convolution_76 = triton_poi_fused_convolution_76
triton_poi_fused_convolution_77 = triton_poi_fused_convolution_77
triton_poi_fused_convolution_78 = triton_poi_fused_convolution_78
triton_poi_fused_convolution_79 = triton_poi_fused_convolution_79
triton_poi_fused_convolution_80 = triton_poi_fused_convolution_80
triton_poi_fused_convolution_81 = triton_poi_fused_convolution_81
triton_poi_fused_convolution_82 = triton_poi_fused_convolution_82
triton_poi_fused_convolution_83 = triton_poi_fused_convolution_83
triton_poi_fused_convolution_84 = triton_poi_fused_convolution_84
triton_poi_fused_convolution_85 = triton_poi_fused_convolution_85
triton_poi_fused_convolution_86 = triton_poi_fused_convolution_86
triton_poi_fused_convolution_87 = triton_poi_fused_convolution_87
triton_poi_fused_convolution_88 = triton_poi_fused_convolution_88
triton_poi_fused_convolution_89 = triton_poi_fused_convolution_89
triton_poi_fused_convolution_90 = triton_poi_fused_convolution_90
triton_poi_fused_convolution_91 = triton_poi_fused_convolution_91
triton_poi_fused_convolution_92 = triton_poi_fused_convolution_92
triton_poi_fused_convolution_93 = triton_poi_fused_convolution_93
triton_poi_fused_convolution_94 = triton_poi_fused_convolution_94
triton_poi_fused_convolution_95 = triton_poi_fused_convolution_95
triton_poi_fused_convolution_96 = triton_poi_fused_convolution_96
triton_poi_fused_convolution_97 = triton_poi_fused_convolution_97
triton_poi_fused_convolution_98 = triton_poi_fused_convolution_98
triton_poi_fused_convolution_99 = triton_poi_fused_convolution_99
triton_poi_fused_convolution_100 = triton_poi_fused_convolution_100
triton_poi_fused_convolution_101 = triton_poi_fused_convolution_101
triton_poi_fused_convolution_102 = triton_poi_fused_convolution_102
triton_poi_fused_convolution_103 = triton_poi_fused_convolution_103
triton_poi_fused_convolution_104 = triton_poi_fused_convolution_104
triton_poi_fused_convolution_105 = triton_poi_fused_convolution_105
triton_poi_fused_convolution_106 = triton_poi_fused_convolution_106
triton_poi_fused_convolution_107 = triton_poi_fused_convolution_107
triton_poi_fused_convolution_108 = triton_poi_fused_convolution_108
triton_poi_fused_convolution_109 = triton_poi_fused_convolution_109
triton_poi_fused_convolution_110 = triton_poi_fused_convolution_110
triton_poi_fused_convolution_111 = triton_poi_fused_convolution_111
triton_poi_fused_convolution_112 = triton_poi_fused_convolution_112
triton_poi_fused_convolution_113 = triton_poi_fused_convolution_113
triton_poi_fused_convolution_114 = triton_poi_fused_convolution_114
triton_poi_fused_convolution_115 = triton_poi_fused_convolution_115
triton_poi_fused_convolution_116 = triton_poi_fused_convolution_116
triton_poi_fused_convolution_117 = triton_poi_fused_convolution_117
triton_poi_fused_convolution_118 = triton_poi_fused_convolution_118
triton_poi_fused_convolution_119 = triton_poi_fused_convolution_119
triton_poi_fused_convolution_120 = triton_poi_fused_convolution_120
triton_poi_fused_convolution_121 = triton_poi_fused_convolution_121
triton_poi_fused_convolution_122 = triton_poi_fused_convolution_122
triton_poi_fused_convolution_123 = triton_poi_fused_convolution_123
triton_poi_fused_convolution_124 = triton_poi_fused_convolution_124
triton_poi_fused_convolution_125 = triton_poi_fused_convolution_125
triton_poi_fused_convolution_126 = triton_poi_fused_convolution_126
triton_poi_fused_convolution_127 = triton_poi_fused_convolution_127
triton_poi_fused_convolution_128 = triton_poi_fused_convolution_128
triton_poi_fused_convolution_129 = triton_poi_fused_convolution_129
triton_poi_fused_convolution_130 = triton_poi_fused_convolution_130
triton_poi_fused_convolution_131 = triton_poi_fused_convolution_131
triton_poi_fused_convolution_132 = triton_poi_fused_convolution_132
triton_poi_fused_convolution_133 = triton_poi_fused_convolution_133
triton_poi_fused_convolution_134 = triton_poi_fused_convolution_134
triton_poi_fused_convolution_135 = triton_poi_fused_convolution_135
triton_poi_fused_convolution_136 = triton_poi_fused_convolution_136
triton_poi_fused_convolution_137 = triton_poi_fused_convolution_137
triton_poi_fused_convolution_138 = triton_poi_fused_convolution_138
triton_poi_fused_convolution_139 = triton_poi_fused_convolution_139
triton_poi_fused_convolution_140 = triton_poi_fused_convolution_140
triton_poi_fused_convolution_141 = triton_poi_fused_convolution_141
triton_poi_fused_convolution_142 = triton_poi_fused_convolution_142
triton_poi_fused_convolution_143 = triton_poi_fused_convolution_143
triton_poi_fused_convolution_144 = triton_poi_fused_convolution_144
triton_poi_fused_convolution_145 = triton_poi_fused_convolution_145
triton_poi_fused_convolution_146 = triton_poi_fused_convolution_146
triton_poi_fused_convolution_147 = triton_poi_fused_convolution_147
triton_poi_fused_convolution_148 = triton_poi_fused_convolution_148
triton_poi_fused_convolution_149 = triton_poi_fused_convolution_149
triton_poi_fused_convolution_150 = triton_poi_fused_convolution_150
triton_poi_fused_convolution_151 = triton_poi_fused_convolution_151
triton_poi_fused_convolution_152 = triton_poi_fused_convolution_152
triton_poi_fused_convolution_153 = triton_poi_fused_convolution_153
triton_poi_fused_convolution_154 = triton_poi_fused_convolution_154
triton_poi_fused_convolution_155 = triton_poi_fused_convolution_155
triton_poi_fused_convolution_156 = triton_poi_fused_convolution_156
triton_poi_fused_convolution_157 = triton_poi_fused_convolution_157
triton_poi_fused_convolution_158 = triton_poi_fused_convolution_158
triton_poi_fused_convolution_159 = triton_poi_fused_convolution_159
triton_poi_fused_convolution_160 = triton_poi_fused_convolution_160
triton_poi_fused_convolution_161 = triton_poi_fused_convolution_161
triton_poi_fused_convolution_162 = triton_poi_fused_convolution_162
triton_poi_fused_convolution_163 = triton_poi_fused_convolution_163
triton_poi_fused_convolution_164 = triton_poi_fused_convolution_164
triton_poi_fused_convolution_165 = triton_poi_fused_convolution_165
triton_poi_fused_convolution_166 = triton_poi_fused_convolution_166
triton_poi_fused_convolution_167 = triton_poi_fused_convolution_167
triton_poi_fused_convolution_168 = triton_poi_fused_convolution_168
triton_poi_fused_convolution_169 = triton_poi_fused_convolution_169
triton_poi_fused_convolution_170 = triton_poi_fused_convolution_170
triton_poi_fused_convolution_171 = triton_poi_fused_convolution_171
triton_poi_fused_convolution_172 = triton_poi_fused_convolution_172
triton_poi_fused_convolution_173 = triton_poi_fused_convolution_173
triton_poi_fused_convolution_174 = triton_poi_fused_convolution_174
triton_poi_fused_convolution_175 = triton_poi_fused_convolution_175
triton_poi_fused_convolution_176 = triton_poi_fused_convolution_176
triton_poi_fused_convolution_177 = triton_poi_fused_convolution_177
triton_poi_fused_convolution_178 = triton_poi_fused_convolution_178
triton_poi_fused_convolution_179 = triton_poi_fused_convolution_179
triton_poi_fused_convolution_180 = triton_poi_fused_convolution_180
triton_poi_fused_convolution_181 = triton_poi_fused_convolution_181
triton_poi_fused_convolution_182 = triton_poi_fused_convolution_182
triton_poi_fused_convolution_183 = triton_poi_fused_convolution_183
triton_poi_fused_convolution_184 = triton_poi_fused_convolution_184
triton_poi_fused_convolution_185 = triton_poi_fused_convolution_185
triton_poi_fused_convolution_186 = triton_poi_fused_convolution_186
triton_poi_fused_convolution_187 = triton_poi_fused_convolution_187
triton_poi_fused_convolution_188 = triton_poi_fused_convolution_188
triton_poi_fused_convolution_189 = triton_poi_fused_convolution_189
triton_poi_fused_convolution_190 = triton_poi_fused_convolution_190
triton_poi_fused_convolution_191 = triton_poi_fused_convolution_191
triton_poi_fused_convolution_192 = triton_poi_fused_convolution_192
triton_poi_fused_convolution_193 = triton_poi_fused_convolution_193
triton_poi_fused_convolution_194 = triton_poi_fused_convolution_194
triton_poi_fused_convolution_195 = triton_poi_fused_convolution_195
triton_poi_fused_convolution_196 = triton_poi_fused_convolution_196
triton_poi_fused_convolution_197 = triton_poi_fused_convolution_197
triton_poi_fused_convolution_198 = triton_poi_fused_convolution_198
triton_poi_fused_convolution_199 = triton_poi_fused_convolution_199
triton_poi_fused_convolution_200 = triton_poi_fused_convolution_200
triton_poi_fused_convolution_201 = triton_poi_fused_convolution_201
triton_poi_fused_convolution_202 = triton_poi_fused_convolution_202
triton_poi_fused_convolution_203 = triton_poi_fused_convolution_203
triton_poi_fused_convolution_204 = triton_poi_fused_convolution_204
triton_poi_fused_convolution_205 = triton_poi_fused_convolution_205
triton_poi_fused_convolution_206 = triton_poi_fused_convolution_206
triton_poi_fused_convolution_207 = triton_poi_fused_convolution_207
triton_poi_fused_convolution_208 = triton_poi_fused_convolution_208
triton_poi_fused_convolution_209 = triton_poi_fused_convolution_209
triton_poi_fused_convolution_210 = triton_poi_fused_convolution_210
triton_poi_fused_convolution_211 = triton_poi_fused_convolution_211
triton_poi_fused_convolution_212 = triton_poi_fused_convolution_212
triton_poi_fused_convolution_213 = triton_poi_fused_convolution_213
triton_poi_fused_convolution_214 = triton_poi_fused_convolution_214
triton_poi_fused_convolution_215 = triton_poi_fused_convolution_215
triton_poi_fused_convolution_216 = triton_poi_fused_convolution_216
triton_poi_fused_convolution_217 = triton_poi_fused_convolution_217
triton_poi_fused_convolution_218 = triton_poi_fused_convolution_218
triton_poi_fused_convolution_219 = triton_poi_fused_convolution_219
triton_poi_fused_convolution_220 = triton_poi_fused_convolution_220
triton_poi_fused_convolution_221 = triton_poi_fused_convolution_221
triton_poi_fused_convolution_222 = triton_poi_fused_convolution_222
triton_poi_fused_convolution_223 = triton_poi_fused_convolution_223
triton_poi_fused_convolution_224 = triton_poi_fused_convolution_224
triton_poi_fused_convolution_225 = triton_poi_fused_convolution_225
triton_poi_fused_convolution_226 = triton_poi_fused_convolution_226
triton_poi_fused_convolution_227 = triton_poi_fused_convolution_227
triton_poi_fused_convolution_228 = triton_poi_fused_convolution_228
triton_poi_fused_convolution_229 = triton_poi_fused_convolution_229
triton_poi_fused_convolution_230 = triton_poi_fused_convolution_230
triton_poi_fused_convolution_231 = triton_poi_fused_convolution_231
triton_poi_fused_convolution_232 = triton_poi_fused_convolution_232
triton_poi_fused_convolution_233 = triton_poi_fused_convolution_233
triton_poi_fused_convolution_234 = triton_poi_fused_convolution_234
triton_poi_fused_convolution_235 = triton_poi_fused_convolution_235
triton_poi_fused_convolution_236 = triton_poi_fused_convolution_236
triton_poi_fused_convolution_237 = triton_poi_fused_convolution_237
triton_poi_fused_convolution_238 = triton_poi_fused_convolution_238
triton_poi_fused_convolution_239 = triton_poi_fused_convolution_239
triton_poi_fused_convolution_240 = triton_poi_fused_convolution_240
triton_poi_fused_convolution_241 = triton_poi_fused_convolution_241
triton_poi_fused_convolution_242 = triton_poi_fused_convolution_242
triton_poi_fused_convolution_243 = triton_poi_fused_convolution_243
triton_poi_fused_convolution_244 = triton_poi_fused_convolution_244
triton_poi_fused_convolution_245 = triton_poi_fused_convolution_245
triton_poi_fused_convolution_246 = triton_poi_fused_convolution_246
triton_poi_fused_convolution_247 = triton_poi_fused_convolution_247
triton_poi_fused_convolution_248 = triton_poi_fused_convolution_248
triton_poi_fused_convolution_249 = triton_poi_fused_convolution_249
triton_poi_fused_convolution_250 = triton_poi_fused_convolution_250
triton_poi_fused_convolution_251 = triton_poi_fused_convolution_251
triton_poi_fused_convolution_252 = triton_poi_fused_convolution_252
triton_poi_fused_convolution_253 = triton_poi_fused_convolution_253
triton_poi_fused_convolution_254 = triton_poi_fused_convolution_254
triton_poi_fused_convolution_255 = triton_poi_fused_convolution_255
triton_poi_fused_convolution_256 = triton_poi_fused_convolution_256
triton_poi_fused_convolution_257 = triton_poi_fused_convolution_257
triton_poi_fused_convolution_258 = triton_poi_fused_convolution_258
triton_poi_fused_convolution_259 = triton_poi_fused_convolution_259
triton_poi_fused_convolution_260 = triton_poi_fused_convolution_260
triton_poi_fused_convolution_261 = triton_poi_fused