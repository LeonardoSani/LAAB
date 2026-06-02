import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Convolutional decoder — symmetric mirror of Encoder (NLSD Table 4 / Appendix D).

    Architecture:
        Linear(latent_dim -> 8d * h * h)            -- unproject
        Reshape to (8d, h, h)
        ConvTranspose2d(8d -> 4d, 3x3, stride=2, pad=1) -> ReLU
        ConvTranspose2d(4d -> 2d, 3x3, stride=2, pad=1) -> ReLU
        ConvTranspose2d(2d -> d,  3x3, stride=2, pad=1) -> ReLU
        ConvTranspose2d(d  -> 1,  3x3, stride=2, pad=1)  -- no output activation

    For MNIST (1x28x28, d=32):
        Encoder spatial sizes: 28 -> 14 -> 7 -> 4 -> 2
        Decoder spatial sizes:  2 ->  4 -> 7 -> 14 -> 28
        output_padding per layer: [1, 0, 1, 1]
          (satisfies: target = 2 * input - 1 + output_padding)
    """

    def __init__(self, latent_dim: int = 256, channel_base: int = 32, out_channels: int = 1, img_size: int = 28):
        super().__init__()
        d = channel_base

        # Reproduce encoder spatial sizes to derive output_padding per layer
        enc_sizes = [img_size]
        h = img_size
        for _ in range(4):
            h = (h + 2 * 1 - 3) // 2 + 1
            enc_sizes.append(h)
        # enc_sizes = [28, 14, 7, 4, 2]

        enc_h = enc_sizes[-1]  # 2

        # output_padding[i] makes ConvTranspose exactly invert the i-th encoder conv.
        # ConvTranspose output size = 2 * input - 1 + output_padding  (for k=3, s=2, p=1)
        def op(layer_idx: int) -> int:
            inp = enc_sizes[4 - layer_idx]
            tgt = enc_sizes[3 - layer_idx]
            return tgt - (2 * inp - 1)

        op0, op1, op2, op3 = op(0), op(1), op(2), op(3)
        # MNIST: op0=1, op1=0, op2=1, op3=1

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
        x = self.deconv4(x)                # no output activation
        return x
