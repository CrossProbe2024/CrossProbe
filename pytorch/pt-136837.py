import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

inp = (torch.rand(10, 32, 32, 1) > 0.5).to(torch.float32)
print(inp.shape)

class MaskedConv2dA(torch.nn.Conv2d):
    def __init__(self, *, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        mask = torch.zeros_like(self.weight)

        mask[:, :, :kernel_size // 2, :] = 1
        mask[:, :, kernel_size // 2, :kernel_size // 2] = 1
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data *= self.mask
        return super().forward(x)


class MaskedConv2dB(torch.nn.Conv2d):
    def __init__(self, *, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        mask = torch.zeros_like(self.weight)

        mask[:, :, :kernel_size // 2, :] = 1
        mask[:, :, kernel_size // 2, :kernel_size // 2] = 1
        mask[:, :, kernel_size // 2, kernel_size // 2] = 1
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data *= self.mask
        return super().forward(x)

@torch.compile
class PixelCNN(torch.nn.Module):
    def __init__(self, num_channels: int, num_colors: int, H: int, W: int, n_layers: int = 5) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_colors = num_colors
        self.H = H
        self.W = W

        kernel_size = 7
        padding = (kernel_size - 1) // 2
        # 1 7x7 Mask A
        layers = [
            MaskedConv2dA(in_channels=self.num_channels, out_channels=64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        ]
        # 5 7x7 Mask B
        for _ in range(n_layers):
          layers.extend(
              [
                  MaskedConv2dB(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding),
                  nn.ReLU(),
              ]
          )
        # 2 1x1 Conv
        layers.extend(
            [
                nn.Conv2d(64, 64, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, num_channels*num_colors, kernel_size=1, padding=0),
            ]
        )
        self.model = nn.Sequential(*layers)


    @staticmethod
    def loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat.permute(0, 4, 1, 2, 3) # (B, H, W, C, K) -> (B, K, H, W, C)
        # y: (B, H, W, C)
        return F.cross_entropy(y_hat, y.long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        # [0, N_c] -> [0, 1] -> [-1, 1]
        x = 2.0*(x.float() / self.num_colors) - 1.0
        x = self.model(x).permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        return x.view(batch_size, self.H, self.W, self.num_channels, self.num_colors) # (B, H, W, C*K) -> (B, H, W, C, K)

    def sample(self, num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            samples = torch.zeros(num_samples, self.H, self.W, self.num_channels).to(device)
            for i in tqdm(range(self.H), desc="Heights"):
                for j in tqdm(range(self.W), desc="Widths"):
                    for k in range(self.num_channels):
                        logits = self.forward(samples)[:, i, j, k, :] # (B, H, W, C, K) -> (B, K)
                        prob = torch.softmax(logits, dim=-1)  # (B, K)
                        samples[:, i, j, k] = torch.multinomial(prob, 1).squeeze(1)  # (B, K) -> (B,)
            return samples.detach().cpu().numpy()


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