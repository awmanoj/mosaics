"""Draw the final tiling (paper sec 5) and a few debug views."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def region_colours(labels: np.ndarray, rgb: np.ndarray, sites: np.ndarray, mode: str) -> np.ndarray:
    """Per-site RGB colour in [0, 1]: mean over the Voronoi region, or a point sample at the centre."""
    n = len(sites)
    h, w, _ = rgb.shape
    xi = np.clip(sites[:, 0].astype(int), 0, w - 1)
    yi = np.clip(sites[:, 1].astype(int), 0, h - 1)
    point = rgb[yi, xi]
    if mode == "point":
        return point
    flat = labels.ravel()
    valid = flat >= 0
    lab = flat[valid]
    cnt = np.bincount(lab, minlength=n)
    mean = np.stack(
        [np.bincount(lab, weights=rgb[..., c].ravel()[valid], minlength=n) for c in range(3)], axis=1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = mean / cnt[:, None]
    mean[cnt == 0] = point[cnt == 0]
    return mean


def tile_corners(cx, cy, phi, side, aspect):
    """Corners of a rectangle of area side^2, elongated by `aspect` along the edge direction."""
    a = side / (2 * np.sqrt(aspect))   # half-extent along the gradient (u)
    b = side * np.sqrt(aspect) / 2     # half-extent along the edge (v)
    c, s = np.cos(phi), np.sin(phi)
    pts = []
    for u, v in ((-a, -b), (a, -b), (a, b), (-a, b)):
        pts.append((cx + u * c - v * s, cy + u * s + v * c))
    return pts


def render_tiles(
    sites: np.ndarray,
    angles: np.ndarray,
    sides: np.ndarray,
    colours: np.ndarray,
    out_size: tuple[int, int],
    scale: float,
    aspect: float = 1.0,
    grout=(40, 40, 40),
    supersample: int = 2,
    bevel: float = 0.0,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Paint one rotated square per site onto a grout-coloured canvas, with supersampled AA."""
    W, H = out_size
    S = max(1, int(supersample))
    img = Image.new("RGB", (W * S, H * S), tuple(int(v) for v in grout))
    draw = ImageDraw.Draw(img)
    if jitter > 0 and rng is not None:
        colours = colours * (1 + jitter * (rng.random((len(sites), 1)) - 0.5))
    colours = np.clip(colours * 255, 0, 255).astype(int)
    f = scale * S
    for (x, y), phi, side, col in zip(sites, angles, sides, colours):
        pts = tile_corners(x * f, y * f, phi, side * f, aspect)
        fill = tuple(int(v) for v in col)
        if bevel > 0:
            outline = tuple(int(v * (1 - 0.6 * bevel)) for v in col)
            draw.polygon(pts, fill=fill, outline=outline, width=max(1, int(S * 0.6 * bevel + 0.5)))
        else:
            draw.polygon(pts, fill=fill)
    if S > 1:
        img = img.resize((W, H), Image.LANCZOS)
    return img


def render_voronoi(labels: np.ndarray, colours: np.ndarray, out_size: tuple[int, int], grout=(40, 40, 40)) -> Image.Image:
    """Haeberli-style output: every Voronoi region filled with its colour (paper fig 12b)."""
    table = np.vstack([np.clip(colours * 255, 0, 255), [grout]]).astype(np.uint8)
    img = Image.fromarray(table[labels])  # label -1 indexes the appended grout row
    return img.resize(out_size, Image.NEAREST)


# ---------------------------------------------------------------- debug views

def field_image(phi: np.ndarray, edges: np.ndarray | None = None) -> Image.Image:
    """Hue encodes the tile orientation (period 90 degrees); edges drawn white."""
    hue = ((phi + np.pi / 4) / (np.pi / 2)) % 1.0
    # vectorised HSV->RGB with s=0.8, v=0.9
    s, v = 0.8, 0.9
    hh = hue * 6
    i = np.floor(hh).astype(int) % 6
    fr = hh - np.floor(hh)
    p, q, t = v * (1 - s), v * (1 - s * fr), v * (1 - s * (1 - fr))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    out = np.stack([r, g, b], axis=-1)
    if edges is not None:
        out[edges] = 1.0
    return Image.fromarray((out * 255).astype(np.uint8))


def labels_image(labels: np.ndarray, sites: np.ndarray, seed: int = 0) -> Image.Image:
    """Random colour per region, white dot at each site (like paper fig 7)."""
    rng = np.random.default_rng(seed)
    n = len(sites)
    table = np.vstack([rng.integers(40, 230, (n, 3)), [[0, 0, 0]]]).astype(np.uint8)
    img = Image.fromarray(table[labels])
    d = ImageDraw.Draw(img)
    for x, y in sites:
        d.point((x, y), fill=(255, 255, 255))
    return img
