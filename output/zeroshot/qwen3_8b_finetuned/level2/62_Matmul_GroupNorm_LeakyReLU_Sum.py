x = self.fc(x)
x = self.gn(x)
x = self.leaky_relu(x)
x = x + x