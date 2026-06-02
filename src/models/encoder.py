import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional encoder (NLSD Table 4 / Appendix D).

    Architecture:
        Conv2d(1   -> d,   3x3, stride=2, pad=1) -> ReLU
        Conv2d(d   -> 2d,  3x3, stride=2, pad=1) -> ReLU
        Conv2d(2d  -> 4d,  3x3, stride=2, pad=1) -> ReLU
        Conv2d(4d  -> 8d,  3x3, stride=2, pad=1) -> ReLU
        Flatten
        Linear(8d * h * h -> latent_dim)          -- no activation at bottleneck

    For MNIST (1x28x28, d=32):
        Spatial sizes: 28 -> 14 -> 7 -> 4 -> 2
        Flat size after conv: 256 * 2 * 2 = 1024
        Bottleneck: 1024 -> 256
    """

    def __init__(self, latent_dim: int = 256, channel_base: int = 32, in_channels: int = 1, img_size: int = 28):
        super().__init__()
        d = channel_base

        self.conv1 = nn.Conv2d(in_channels, d,     kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d,           2 * d, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2 * d,       4 * d, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(4 * d,       8 * d, kernel_size=3, stride=2, padding=1)
        self.relu  = nn.ReLU()

        # Compute spatial size after 4 stride-2 convs: floor((h + 2p - k) / s) + 1
        h = img_size
        for _ in range(4):
            h = (h + 2 * 1 - 3) // 2 + 1   # h=28 -> 14 -> 7 -> 4 -> 2

        self.flatten  = nn.Flatten()
        self.project  = nn.Linear(8 * d * h * h, latent_dim)  # no activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        z = self.project(x)
        return z
