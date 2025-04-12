import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # Import tqdm for progress bars

# Example input tensor
inp = (torch.rand(10, 32, 32, 1) > 0.5).to(torch.float32)
print(inp.shape)

class MaskedConv2dA(torch.nn.Conv2d):
    """
    Masked 2D Convolution for PixelCNN - Mask A
    """
    def __init__(self, *, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        mask = torch.zeros_like(self.weight)

        mask[:, :, :kernel_size // 2, :] = 1
        mask[:, :, kernel_Size // 2, :kernel_Size // 2] = 1
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data *= self.mask  # Apply mask to weights
        return super().forward(x)


class MaskedConv2dB(torch.nn.Conv2d):
    """
    Masked 2D Convolution for PixelCNN - Mask B
    """
    def __init__(self, *, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        mask = torch.zeros_like(self.weight)

        mask[:, :, :kernel_size // 2, :] = 1
        mask[:, :, kernel_size // 2, :kernel_size // 2] = 1
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data *= self.mask  # Apply mask to weights
        return super().forward(x)


class PixelCNN(nn.Module):
    """
    PixelCNN model for image generation.
    """
    def __init__(self, H: int, W: int, num_channels: int, num_colors: int):
        super().__init__()
        self.H = H
        self.W = W
        self.num_channels = num_channels
        self.num_colors = num_colors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PixelCNN model.
        """
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        # [0, N_c] -> [0, 1] -> [-1, 1]
        x = 2.0 * (x.float() / self.num_colors) - 1.0
        x = self.model(x).permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        return x.view(batch_size, self.H, self.W, self.num_channels, self.num_colors)  # (B, H, W, C*K) -> (B, H, W, C, K)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the PixelCNN model.
        """
        with torch.no_grad():
            samples = torch.zeros(num_samples, self.H, self.W, self.num_channels).to(device)
            for i in tqdm(range(self.H), desc="Heights"):
                for j in tqdm(range(self.W), desc="Widths"):
                    for k in range(self.num_channels):
                        logits = self.forward(samples)[:, i, j, k, :]  # (B, H, W, C, K) -> (B, K)
                        prob = torch.softmax(logits, dim=-1)  # (B, K)
                        samples[:, i, j, k] = torch.multinomial(prob, 1).squeeze(1)  # (B, K) -> (B,)
            return samples.detach().cpu().numpy()

    def model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the PixelCNN model architecture.
        """
        # Simple example model - Replace with more complex architecture
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    # Define layers
    conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
    conv3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
pixel_cnn = PixelCNN(H=32, W=32, num_channels=1, num_colors=2).to(device)
print(pixel_cnn((torch.rand(10, 32, 32, 1) > 0.5).to(torch.float32)).shape)