import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Mirror of Encoder: linear unproject + four stride-2 ConvTranspose2d.
    MNIST d=32: 2->4->7->14->28, output_padding inverts the encoder."""

    def __init__(self, latent_dim: int = 256, channel_base: int = 32, out_channels: int = 1, img_size: int = 28):
        super().__init__()
        d = channel_base

        # encoder spatial sizes, to derive output_padding per layer
        enc_sizes = [img_size]
        h = img_size
        for _ in range(4):
            h = (h + 2 * 1 - 3) // 2 + 1
            enc_sizes.append(h)
        enc_h = enc_sizes[-1]

        # output_padding to invert each encoder conv: out = 2*in - 1 + op
        def op(layer_idx: int) -> int:
            inp = enc_sizes[4 - layer_idx]
            tgt = enc_sizes[3 - layer_idx]
            return tgt - (2 * inp - 1)

        op0, op1, op2, op3 = op(0), op(1), op(2), op(3)

        self.unproject = nn.Linear(latent_dim, 8 * d * enc_h * enc_h)
        self._enc_h  = enc_h
        self._enc_ch = 8 * d

        self.deconv1 = nn.ConvTranspose2d(8 * d, 4 * d,       kernel_size=3, stride=2, padding=1, output_padding=op0)
        self.deconv2 = nn.ConvTranspose2d(4 * d, 2 * d,       kernel_size=3, stride=2, padding=1, output_padding=op1)
        self.deconv3 = nn.ConvTranspose2d(2 * d, d,           kernel_size=3, stride=2, padding=1, output_padding=op2)
        self.deconv4 = nn.ConvTranspose2d(d,     out_channels, kernel_size=3, stride=2, padding=1, output_padding=op3)
        self.relu    = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.unproject(z)
        x = x.view(x.size(0), self._enc_ch, self._enc_h, self._enc_h)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x
