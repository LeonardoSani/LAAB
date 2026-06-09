"""Count unique attractors by cosine-threshold clustering."""
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class UniquenessStats:
    n_total: int
    n_unique: int
    threshold: float
    cluster_sizes: list[int]          # component sizes, descending
    components: list[list[int]]       # indices per component

    def __str__(self):
        size_counts = Counter(self.cluster_sizes)
        size_str = "  ".join(f"{v}×(size {k})" for k, v in sorted(size_counts.items()))
        return (
            f"Unique attractors: {self.n_unique}/{self.n_total} "
            f"(cos > {self.threshold})  |  {size_str}"
        )


def count_unique_attractors(
    A: torch.Tensor,
    threshold: float,
) -> UniquenessStats:
    """Unique attractors = connected components of cos(z_i, z_j) > threshold. A: (N, k)."""
    N = A.size(0)
    A_norm = F.normalize(A.float().cpu(), p=2, dim=1)
    sim = A_norm @ A_norm.T

    adj = (sim > threshold)
    adj.fill_diagonal_(False)

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
