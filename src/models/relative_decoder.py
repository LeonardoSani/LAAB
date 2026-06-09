import torch

from src.models.decoder import Decoder


class RelativeDecoder(Decoder):
    """AE decoder taking an N-dim relative code in [-1,1]. N=k=256 keeps the
    input layer unchanged; its weights absorb the code-vs-z scale mismatch."""

    def __init__(self, N: int = 256, channel_base: int = 32, out_channels: int = 1, img_size: int = 28):
        super().__init__(latent_dim=N, channel_base=channel_base, out_channels=out_channels, img_size=img_size)

    @classmethod
    def from_checkpoint(cls, ckpt_path, cfg, device) -> "RelativeDecoder":
        """Build from AEConfig and load a checkpoint."""
        dec = cls(
            N=cfg.latent_dim, channel_base=cfg.channel_base,
            out_channels=cfg.in_channels, img_size=cfg.img_size,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        dec.load_state_dict(ckpt["model_state"])
        dec.eval()
        return dec
