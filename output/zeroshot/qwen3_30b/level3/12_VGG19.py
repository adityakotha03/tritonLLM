@triton.jit
def conv2d_kernel(
    x_ptr,      # input pointer: [N, C_in, H, W]
    w_ptr,      # weight pointer: [C_out, C_in, 3, 3]
    b_ptr,      # bias pointer: [C_out]
    out_ptr,    # output pointer: [N, C_out, H, W]
    n, c_in, h, w, c_out,  # dimensions
    stride_h, stride_w, padding_h, padding_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):