x = self.batch_norm(x)
x = x - torch.mean(x, dim=(2,3,4), keepdim=True)