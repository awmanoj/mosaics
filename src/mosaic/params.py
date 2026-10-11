from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MosaicParams:
    """All knobs of the mosaic pipeline. Defaults follow the paper where it gives values."""

    # --- tile count / size ---
    tiles: int = 2000                 # number of tiles N (ignored if tile_size is set)
    tile_size: float | None = None    # tile side in working-resolution pixels (overrides tiles)
    fill: float = 0.8                 # beta: packing factor, d = sqrt(beta*h*w/n)  (paper: 0.8)
    aspect: float = 1.0               # alpha: tile elongation along the edge direction (paper sec 5.3)

    # --- working resolution ---
    max_size: int = 1024              # longest side used for computation
    out_max_size: int | None = None   # longest side of the output; None keeps the input size

    # --- Lloyd iterations ---
    iters: int = 25                   # iterations with edges drawn (edge avoidance, paper sec 6)
    relax_iters: int = 3              # final iterations without edges to close the gaps
    neighbours: int = 16              # candidate sites per pixel when evaluating the rotated metric
    seed: int = 0

    # --- edge detection (Canny) ---
    edge_sigma: float = 2.0
    edge_low: float = 0.1
    edge_high: float = 0.3
    min_edge_len: float = 2.0         # drop edge fragments shorter than this many tile widths
    edge_width: float = 0.3           # thickness of the drawn edge lines, as a fraction of tile size

    # --- variable tile size (paper sec 5.2) ---
    detail_scale: float = 1.0         # smallest tile size near edges relative to base (1.0 = uniform)
    detail_radius: float = 3.0        # distance (in tiles) over which size ramps back up to base

    # --- appearance ---
    color: str = "mean"               # "mean": average over the tile's region, "point": centre sample
    style: str = "tiles"              # "tiles": rotated squares, "voronoi": filled regions (Haeberli)
    grout: tuple[int, int, int] = (40, 40, 40)
    supersample: int = 2              # anti-aliasing factor for tile rendering
    bevel: float = 0.0                # 0..1 strength of a darker tile outline
    jitter: float = 0.0               # 0..1 random per-tile brightness variation
