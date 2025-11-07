def forward(self, predictions, targets):
    return torch.mean((predictions - targets) ** 2)