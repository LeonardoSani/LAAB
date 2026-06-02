import torch

from src.models.decoder import Decoder


class RelativeDecoder(Decoder):
    """
    Relative decoder D_r (§7): architecturally identical to the AE decoder D_1.

    Accepts an N-dimensional relative representation r(z) ∈ [-1,1]^N as input.
    With N = k = 256 the linear input layer is dimensionally identical to D_1's.
    No input normalization in the first baseline; the first linear layer absorbs
    any scale or distribution mismatch between r(z) and the z that D_1 was trained on.
    """

    def __init__(self, N: int = 256, channel_base: int = 32, out_channels: int = 1, img_size: int = 28):
        super().__init__(latent_dim=N, channel_base=channel_base, out_channels=out_channels, img_size=img_size)

    @classmethod
    def from_checkpoint(cls, ckpt_path, cfg, device) -> "RelativeDecoder":
        """Build from an AEConfig (N = latent_dim) and load a trained checkpoint."""
        dec = cls(
            N=cfg.latent_dim, channel_base=cfg.channel_base,
            out_channels=cfg.in_channels, img_size=cfg.img_size,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        dec.load_state_dict(ckpt["model_state"])
        dec.eval()
        return dec
