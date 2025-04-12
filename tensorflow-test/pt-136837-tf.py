import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Define MaskedConv2DA class
class MaskedConv2DA(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding="same"):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        # self.kernel_size = kernel_size
        # self.mask = None

    def forward(self, x):
        # Create mask during forward pass
        mask = torch.zeros(self.weight.shape, dtype=torch.float32, device=self.weight.device)
        center = self.kernel_size[0] // 2
        mask[:, :center, :, :] = 1
        mask[:, center, :, :] = 1
        # Apply mask to the kernel
        self.weight.data = self.weight.data * mask
        return super().forward(x)

# Define MaskedConv2DB class
class MaskedConv2DB(MaskedConv2DA):
    def forward(self, x):
        # Create mask during forward pass
        mask = torch.zeros(self.weight.shape, dtype=torch.float32, device=self.weight.device)
        center = self.kernel_size[0] // 2
        mask[:, :center, :, :] = 1
        mask[:, center, :, :] = 1
        mask[:, center, center, :] = 1  # Corrected index to include the center pixel
        # Apply mask to the kernel
        self.weight.data = self.weight.data * mask
        return super().forward(x)

# Define PixelCNN class
class PixelCNN(nn.Module):
    def __init__(self, H, W, num_channels, num_colors):
        super().__init__()
        self.H = H
        self.W = W
        self.num_channels = num_channels
        self.num_colors = num_colors

        # Define layers
        self.conv1 = MaskedConv2DA(num_channels, 64, 7, padding="same")
        self.conv2 = MaskedConv2DB(64, 64, 7, padding="same")
        # Add more layers as needed

    def forward(self, x):
        # x shape: (B, H, W, C)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Add more layers as needed
        return x

    def sample(self, num_samples):
        samples = torch.zeros((num_samples, self.H, self.W, self.num_channels), dtype=torch.float32)
        for i in tqdm(range(self.H), desc="Heights"):
            for j in tqdm(range(self.W), desc="Widths"):
                for k in range(self.num_channels):
                    logits = self.forward(samples)[:, i, j, k, :]  # (B, K)
                    prob = F.softmax(logits, dim=-1)
                    samples[:, i, j, k] = F.categorical(prob, num_samples=1).squeeze()
        return samples.numpy()

    @staticmethod
    def loss(y_hat, y):
        # y_hat shape: (B, H, W, C, K)
        # y shape: (B, H, W, C)
        y_hat = y_hat.permute(0, 4, 1, 2, 3)  # (B, K, H, W, C)
        return F.cross_entropy(y_hat, y, reduction='mean')

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Initialize model
pixel_cnn = PixelCNN(H=32, W=32, num_channels=1, num_colors=2).to(device)

# Sample input
sample_input = torch.rand((10, 32, 32, 1)).to(device)

# Test layers
orig_layer = nn.Conv2d(filters=64, kernel_size=7, padding="same")
test_layer = MaskedConv2DA(in_channels=1, out_channels=64, kernel_size=7, padding="same")

# Forward pass
output = pixel_cnn(sample_input)
# output = test_layer(sample_input)
print(output.shape)