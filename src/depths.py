"""Depth-grid string conventions, shared across experiments and stores.

Finite depths are ints; the attractor depth is float('inf'), encoded on disk
as the literal "inf" (e.g. phi_anchors_s1_tinf.pt). Single source of truth for
the conversions that used to be copy-pasted in every script.
"""
from typing import Union

DepthLike = Union[int, float]


def depth_to_str(t: DepthLike) -> str:
    """Depth -> filename token: float('inf') -> 'inf', else the int."""
    return "inf" if t == float("inf") else str(int(t))


def str_to_depth(s: str) -> DepthLike:
    """Filename token / yaml entry -> depth: 'inf' -> float('inf'), else int."""
    return float("inf") if str(s) == "inf" else int(s)


def parse_depth_grid(raw: list) -> list[DepthLike]:
    """Parse a yaml depth list (ints + 'inf') into typed depths."""
    return [str_to_depth(g) for g in raw]
