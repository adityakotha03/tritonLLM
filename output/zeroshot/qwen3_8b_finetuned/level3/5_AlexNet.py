def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.maxpool2(x)
    x = self.conv3(x)
    x = self.relu3(x)
    x = self.conv4(x)
    x = self.relu4(x)
    x = self.conv5(x)
    x = self.relu5(x)
    x = self.maxpool3(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = triton_relu(x)  # Replaced ReLU after fc1
    x = self.dropout1(x)
    x = self.fc2(x)
    x = triton_relu(x)  # Replaced ReLU after fc2
    x = self.dropout2(x)
    x = self.fc3(x)
    return x