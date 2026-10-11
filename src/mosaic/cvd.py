"""Centroidal Voronoi diagram under a locally rotated square metric (paper sec 3).

The paper renders one square pyramid per site on the GPU and lets the z-buffer pick, for every
pixel, the site with the lowest pyramid height, where (u, v) are the pixel offsets in the site's
frame rotated by the direction field. We do the same on the CPU: a k-d tree proposes the `k`
Euclidean-nearest sites per pixel and the rotated metric picks the winner among them. Lloyd's
method then moves every site to the centroid of its region.

Metric note: the paper writes the pyramid as z = |u| + |v| (Manhattan). The level sets of that
pyramid are diamonds, and Lloyd's method under it settles into square cells whose *sides lie at
45 degrees to the (u, v) axes* (the diamond cell has lower energy, 0.471 A^1.5 vs 0.5 A^1.5).
A square pyramid whose base is aligned with the axes is z = max(|u|, |v|) (Chebyshev), which is
the Manhattan pyramid rotated by 45 degrees; it yields square cells whose sides follow the field,
so the tiles drawn along phi match their cells. We use that form, with the aspect ratio applied
as max(|u|, |v| / alpha) (paper sec 5.3) and the size as a slope factor (paper sec 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

ProgressFn = Callable[[int, int, float], None]


def pixel_centres(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1).astype(np.float32)


def square_metric(dx, dy, phi, size, aspect):
    """Height of the site's square pyramid at offset (dx, dy): max(|u|, |v|/aspect) / size."""
    c, s = np.cos(phi), np.sin(phi)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return np.maximum(np.abs(u), np.abs(v) / aspect) / size


def assign_regions(
    sites: np.ndarray,
    angles: np.ndarray,
    sizes: np.ndarray,
    aspect: float,
    shape: tuple[int, int],
    k: int = 16,
    blocked: np.ndarray | None = None,
    pts: np.ndarray | None = None,
    chunk: int = 262_144,
) -> np.ndarray:
    """Label every pixel with the index of the site whose pyramid is lowest there.

    `blocked` pixels (the drawn edge lines of sec 6) get label -1 so they pull no site.
    """
    h, w = shape
    if pts is None:
        pts = pixel_centres(shape)
    n = len(sites)
    k = min(k, n)
    tree = cKDTree(sites)
    labels = np.empty(h * w, np.int32)
    for start in range(0, h * w, chunk):
        p = pts[start : start + chunk]
        _, idx = tree.query(p, k=k, workers=-1)
        if k == 1:
            idx = idx[:, None]
        dx = p[:, 0:1] - sites[idx, 0].astype(np.float32)
        dy = p[:, 1:2] - sites[idx, 1].astype(np.float32)
        d = square_metric(dx, dy, angles[idx], sizes[idx], aspect)
        best = np.argmin(d, axis=1)
        labels[start : start + chunk] = idx[np.arange(len(p)), best]
    labels = labels.reshape(h, w)
    if blocked is not None:
        labels[blocked] = -1
    return labels


def region_centroids(labels: np.ndarray, n: int, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids (n,2), pixel counts (n,)); centroid is NaN for empty regions."""
    flat = labels.ravel()
    valid = flat >= 0
    lab = flat[valid]
    cnt = np.bincount(lab, minlength=n).astype(np.float64)
    sx = np.bincount(lab, weights=pts[valid, 0], minlength=n)
    sy = np.bincount(lab, weights=pts[valid, 1], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.stack([sx / cnt, sy / cnt], axis=1)
    return c, cnt


def sample_map(m: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of a (H, W) map at floating-point site positions."""
    h, w = m.shape
    xi = np.clip(sites[:, 0].astype(int), 0, w - 1)
    yi = np.clip(sites[:, 1].astype(int), 0, h - 1)
    return m[yi, xi]


def init_sites(n: int, shape: tuple[int, int], size_map: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random sites, rejection-sampled with probability (s_min / s)^2 so small tiles start denser."""
    h, w = shape
    smin = float(size_map.min())
    out = []
    got = 0
    while got < n:
        cand = rng.random((max(2 * (n - got), 64), 2)) * [w, h]
        s = sample_map(size_map, cand)
        keep = rng.random(len(cand)) < (smin / s) ** 2
        cand = cand[keep][: n - got]
        out.append(cand)
        got += len(cand)
    return np.concatenate(out).astype(np.float64)


@dataclass
class LloydState:
    sites: np.ndarray      # (n, 2) x, y
    angles: np.ndarray     # (n,) tile orientation phi at the site
    sizes: np.ndarray      # (n,) relative size factor at the site
    labels: np.ndarray     # (H, W) final region labels (-1 = edge/grout)


def lloyd(
    sites: np.ndarray,
    phi: np.ndarray,
    size_map: np.ndarray,
    aspect: float,
    iters: int,
    relax_iters: int,
    blocked: np.ndarray | None,
    k: int,
    rng: np.random.Generator,
    progress: ProgressFn | None = None,
    tol: float = 0.05,
) -> LloydState:
    """Run Lloyd's method: `iters` passes with the edge lines blocked, then `relax_iters` without."""
    h, w = phi.shape
    pts = pixel_centres((h, w))
    n = len(sites)
    total = iters + relax_iters
    labels = None
    for it in range(total):
        angles = sample_map(phi, sites).astype(np.float32)
        sizes = sample_map(size_map, sites).astype(np.float32)
        use_block = blocked if it < iters else None
        labels = assign_regions(sites, angles, sizes, aspect, (h, w), k, use_block, pts)
        cent, cnt = region_centroids(labels, n, pts)
        empty = cnt == 0
        if empty.any():  # a site swallowed entirely by neighbours or an edge: respawn it
            cent[empty] = rng.random((int(empty.sum()), 2)) * [w, h]
        moved = float(np.max(np.hypot(*(cent - sites).T)))
        sites = cent
        if progress:
            progress(it + 1, total, moved)
        if moved < tol and it >= iters:
            break
    angles = sample_map(phi, sites).astype(np.float32)
    sizes = sample_map(size_map, sites).astype(np.float32)
    labels = assign_regions(sites, angles, sizes, aspect, (h, w), k, blocked, pts)
    return LloydState(sites, angles, sizes, labels)
