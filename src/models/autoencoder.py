import torch
import torch.nn as nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256, channel_base=32, in_channels=1, img_size=28):
        super().__init__()
        self.encoder = Encoder(latent_dim, channel_base, in_channels, img_size)
        self.decoder = Decoder(latent_dim, channel_base, in_channels, img_size)

    @classmethod
    def from_checkpoint(cls, ckpt_path, cfg, device) -> "Autoencoder":
        """Build from AEConfig and load a checkpoint (eval mode)."""
        model = cls(
            latent_dim=cfg.latent_dim, channel_base=cfg.channel_base,
            in_channels=cfg.in_channels, img_size=cfg.img_size,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))
