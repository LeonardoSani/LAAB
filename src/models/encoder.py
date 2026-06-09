import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Four stride-2 Conv2d (d->2d->4d->8d, 3x3, ReLU) + linear bottleneck.
    MNIST d=32: 28->14->7->4->2, flat 1024 -> 256."""

    def __init__(self, latent_dim: int = 256, channel_base: int = 32, in_channels: int = 1, img_size: int = 28):
        super().__init__()
        d = channel_base

        self.conv1 = nn.Conv2d(in_channels, d,     kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d,           2 * d, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2 * d,       4 * d, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(4 * d,       8 * d, kernel_size=3, stride=2, padding=1)
        self.relu  = nn.ReLU()

        h = img_size  # spatial size after 4 stride-2 convs
        for _ in range(4):
            h = (h + 2 * 1 - 3) // 2 + 1

        self.flatten  = nn.Flatten()
        self.project  = nn.Linear(8 * d * h * h, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        z = self.project(x)
        return z
