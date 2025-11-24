x = tl.load(x_ptr + offsets, mask=mask)
sum_val = tl.sum(x * x)