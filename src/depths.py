"""Depths: finite ints, attractor float('inf'), 'inf' on disk."""
from typing import Union

DepthLike = Union[int, float]


def depth_to_str(t: DepthLike) -> str:
    """inf -> 'inf', else str(int)."""
    return "inf" if t == float("inf") else str(int(t))


def str_to_depth(s: str) -> DepthLike:
    """'inf' -> inf, else int."""
    return float("inf") if str(s) == "inf" else int(s)


def parse_depth_grid(raw: list) -> list[DepthLike]:
    """Parse a yaml depth list."""
    return [str_to_depth(g) for g in raw]
