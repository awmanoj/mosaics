"""End-to-end mosaic pipeline. `make_mosaic` is the API the CLI (and later the web app) call."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation

from .cvd import LloydState, init_sites, lloyd
from .edges import detect_edges
from .field import direction_field, size_factor_map
from .params import MosaicParams
from .render import field_image, labels_image, region_colours, render_tiles, render_voronoi


@dataclass
class MosaicResult:
    image: Image.Image          # final mosaic at the input resolution
    working: np.ndarray         # working-resolution RGB float image
    edges: np.ndarray           # thin edge mask (working res)
    phi: np.ndarray             # direction field (working res)
    state: LloydState           # sites, angles, sizes, labels (working res)
    sides: np.ndarray           # tile side per site (working res px)
    colours: np.ndarray         # per-tile RGB in [0, 1]
    scale: float                # output px per working px

    def save_debug(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray((self.working * 255).astype(np.uint8)).save(folder / "working.png")
        Image.fromarray((self.edges * 255).astype(np.uint8)).save(folder / "edges.png")
        field_image(self.phi, self.edges).save(folder / "field.png")
        labels_image(self.state.labels, self.state.sites).save(folder / "voronoi.png")
        render_voronoi(self.state.labels, self.colours, self.image.size).save(folder / "regions.png")


def _working_copy(image: Image.Image, max_size: int) -> tuple[np.ndarray, float]:
    image = image.convert("RGB")
    W, H = image.size
    s = max(W, H) / max_size
    if s > 1:
        image = image.resize((round(W / s), round(H / s)), Image.LANCZOS)
    else:
        s = 1.0
    return np.asarray(image, dtype=np.float32) / 255.0, s


def make_mosaic(
    image: Image.Image,
    params: MosaicParams | None = None,
    edge_mask: np.ndarray | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> MosaicResult:
    """Turn `image` into a mosaic.

    edge_mask: optional boolean (H, W) mask at the *working* resolution (or any size; it is
    resized) of user-chosen edge curves. When None, edges are detected with Canny.
    progress(stage, i, n) is called as work advances.
    """
    p = params or MosaicParams()
    rng = np.random.default_rng(p.seed)
    image = ImageOps.exif_transpose(image)  # honour the EXIF orientation tag of phone photos
    out_size = image.size
    if p.out_max_size and max(out_size) > p.out_max_size:
        f = p.out_max_size / max(out_size)
        out_size = (max(1, round(out_size[0] * f)), max(1, round(out_size[1] * f)))

    rgb, _ = _working_copy(image, p.max_size)
    h, w, _ = rgb.shape
    scale = out_size[0] / w  # output pixels per working pixel

    # --- tile size (sec 5.2): d = sqrt(beta * h * w / n)
    if p.tile_size:
        d = float(p.tile_size)
        n = max(1, int(round(p.fill * h * w / d**2)))
    else:
        n = max(1, int(p.tiles))
        d = float(np.sqrt(p.fill * h * w / n))

    # --- edges (sec 4) and direction field
    if edge_mask is not None:
        if edge_mask.shape != (h, w):
            edge_mask = np.asarray(
                Image.fromarray(edge_mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            ) > 127
        edges = edge_mask.astype(bool)
    else:
        edges = detect_edges(rgb, p.edge_sigma, p.edge_low, p.edge_high, p.min_edge_len * d)
    phi, dist = direction_field(edges, smooth=max(1.0, d / 2))
    size_map = size_factor_map(dist, d, p.detail_scale, p.detail_radius)

    # --- thick edge lines for edge avoidance (sec 6)
    blocked = None
    if edges.any() and p.edge_width > 0:
        r = max(1, int(round(p.edge_width * d / 2)))
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        blocked = binary_dilation(edges, structure=(xx**2 + yy**2) <= r * r)

    # --- Lloyd's method with the rotated Manhattan metric (sec 3)
    sites = init_sites(n, (h, w), size_map, rng)

    def _prog(i, total, moved):
        if progress:
            progress(f"lloyd (max move {moved:.2f}px)", i, total)

    state = lloyd(sites, phi, size_map, p.aspect, p.iters, p.relax_iters, blocked, p.neighbours, rng, _prog)

    # --- per-tile size: mean tile area equals beta * h * w / n
    base = np.sqrt(p.fill * h * w / (n * np.mean(state.sizes**2)))
    sides = base * state.sizes

    # --- colours (sec 5.1) and rendering
    colours = region_colours(state.labels, rgb, state.sites, p.color)
    if p.style == "voronoi":
        img = render_voronoi(state.labels, colours, out_size, p.grout)
    else:
        img = render_tiles(
            state.sites, state.angles, sides, colours, out_size, scale, p.aspect,
            p.grout, p.supersample, p.bevel, p.jitter, rng,
        )
    return MosaicResult(img, rgb, edges, phi, state, sides, colours, scale)
