def triton_conv3d(x, weight, bias, stride=1, padding=0):
    # Convert to BF16
    x_bf16 = x.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16) if bias is not None else None
    # ... call Triton kernel
    # Then convert back to FP32
    return out_fp32