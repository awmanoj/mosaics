"""Direction field phi (paper sec 4): gradient of the distance to the nearest edge curve."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def direction_field(edges: np.ndarray, smooth: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (phi, dist).

    dist[y, x] is the Euclidean distance to the nearest edge pixel (the "mountain" height of the
    paper). phi[y, x] is the orientation of its gradient, i.e. perpendicular to the nearest edge.
    Square tiles have 90-degree symmetry, so the field is smoothed in (cos 4phi, sin 4phi) space,
    which keeps the ridge (where the gradient flips) from producing noise.
    """
    h, w = edges.shape
    if not edges.any():
        return np.zeros((h, w), np.float32), np.full((h, w), np.hypot(h, w), np.float32)

    dist = distance_transform_edt(~edges).astype(np.float32)
    gy, gx = np.gradient(dist)
    phi = np.arctan2(gy, gx)

    mag = np.hypot(gx, gy)
    c = gaussian_filter(np.cos(4 * phi) * mag, smooth)
    s = gaussian_filter(np.sin(4 * phi) * mag, smooth)
    phi = (np.arctan2(s, c) / 4).astype(np.float32)
    return phi, dist


def size_factor_map(dist: np.ndarray, tile: float, detail_scale: float, detail_radius: float) -> np.ndarray:
    """Tile-size multiplier per pixel: detail_scale at edges ramping to 1 over detail_radius tiles."""
    if detail_scale >= 1.0:
        return np.ones_like(dist, dtype=np.float32)
    t = np.clip(dist / (detail_radius * tile), 0, 1)
    t = t * t * (3 - 2 * t)  # smoothstep
    return (detail_scale + (1 - detail_scale) * t).astype(np.float32)
