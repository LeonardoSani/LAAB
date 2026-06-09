import torch
import torch.nn.functional as F


def relative_cosine(Z: torch.Tensor, S: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """cos(Z, S). Z (B,k), S (N,k) -> (B,N) in [-1,1]."""
    Z_norm = F.normalize(Z, p=2, dim=1, eps=eps)
    S_norm = F.normalize(S, p=2, dim=1, eps=eps)
    return Z_norm @ S_norm.T
