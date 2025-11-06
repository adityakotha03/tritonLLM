max_val = tl.max(x, axis=1)
sum_exp = tl.sum(tl.exp(x - max_val), axis=1)