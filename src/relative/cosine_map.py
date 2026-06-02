import torch
import torch.nn.functional as F


def relative_cosine(Z: torch.Tensor, S: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Batched relative cosine representation (RRZS eq. 1).

    r(Z) = Z_norm @ S_norm.T  in R^{B x N}

    Args:
        Z: (B, k) batch of latent embeddings
        S: (N, k) anchor matrix, rows are anchors — detached, no grad
        eps: small value added to L2 norms to prevent division by zero

    Returns:
        (B, N) cosine similarities in [-1, 1]
    """
    Z_norm = F.normalize(Z, p=2, dim=1, eps=eps)
    S_norm = F.normalize(S, p=2, dim=1, eps=eps)
    return Z_norm @ S_norm.T
