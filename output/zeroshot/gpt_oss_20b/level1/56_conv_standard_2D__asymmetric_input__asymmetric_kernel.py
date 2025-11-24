@triton.jit
def conv2d_kernel(
    inp_ptr,
    filt_ptr,
    out_ptr,
    B, C, H, W,
    O, KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
    ):
    program_id = tl.program_id(0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # compute indices for each offset
    # offset -> (b, oc, oh, ow)
    # b = offset // (O * H_out * W_out)
    # rest ...
    mask = offsets < B * O * H_out * W_out
    # compute each coordinate
    batch = offsets // (O * H_out * W_out)
    rem = offsets % (O * H_out * W_out)
    oc = rem // (H_out * W_out)
    oh = (rem // W_out) % H_out
    ow = rem % W_out
    # compute offset in input
    # base input offset for each channel: 
    # We iterate over kernel positions and input channels
    out_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # For each kernel height
    for kh in range(KH):
        for kw in range(KW):
            # compute input positions
            ih = oh * stride_h + kh * dilation_h - pad_h
            iw = ow * stride_w + kw * dilation_w - pad_w
            # mask if within bounds
            valid_mask = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # For each input channel (within group)
            for ic_group in range(C // groups):
                ic = ic_group + groups * (oc % groups)
                # compute offset in input: batch * C * H * W + ic * H * W + ih * W + iw
                # We need to broadcast across the block.
                input_offset = (batch * C * H * W + ic * H * W + ih * W + iw)
                inp = tl.load(inp_ptr + input_offset, mask=valid_mask, other=0.0)
                # filter offset: oc * (C//groups) * KH * KW + ic_group * KH * KW + kh * KW + kw
                filt_offset = (oc * (C // groups) * KH * KW + ic_group * KH * KW + kh * KW + kw)
                filt = tl.load(filt_ptr + filt_offset, mask=mask, other=0.0)
                out_val += inp * filt
    # store out
    out_ptr + offsets, out_val, mask