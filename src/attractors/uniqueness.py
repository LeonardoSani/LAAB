"""Count unique attractors via cosine similarity threshold."""
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class UniquenessStats:
    n_total: int
    n_unique: int
    threshold: float
    cluster_sizes: list[int]          # size of each connected component, descending
    components: list[list[int]]       # indices per component, same order

    def __str__(self):
        size_counts = Counter(self.cluster_sizes)
        size_str = "  ".join(f"{v}×(size {k})" for k, v in sorted(size_counts.items()))
        return (
            f"Unique attractors: {self.n_unique}/{self.n_total} "
            f"(cos > {self.threshold})  |  {size_str}"
        )


def count_unique_attractors(
    A: torch.Tensor,
    threshold: float = 0.99,
) -> UniquenessStats:
    """
    Count unique attractors using connected components on the cosine similarity graph.

    Two attractors are in the same component (considered equal) if
    cos(z_i, z_j) > threshold.  Components are found via BFS.

    Args:
        A:         (N, k) attractor latent points
        threshold: cosine similarity threshold (default 0.99)

    Returns:
        UniquenessStats with component count, sizes, and member indices
    """
    N = A.size(0)
    A_norm = F.normalize(A.float().cpu(), p=2, dim=1)
    sim = A_norm @ A_norm.T                  # (N, N) full pairwise cosine sim

    # Adjacency matrix: i~j iff cos > threshold, excluding self-loops
    adj = (sim > threshold)
    adj.fill_diagonal_(False)

    # BFS connected components
    visited = torch.zeros(N, dtype=torch.bool)
    components: list[list[int]] = []

    for start in range(N):
        if visited[start]:
            continue
        component: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            neighbors = adj[node].nonzero(as_tuple=True)[0].tolist()
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(component)

    components.sort(key=len, reverse=True)
    sizes = [len(c) for c in components]

    return UniquenessStats(
        n_total=N,
        n_unique=len(components),
        threshold=threshold,
        cluster_sizes=sizes,
        components=components,
    )
