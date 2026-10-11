"""Command line: `mosaic INPUT -o OUTPUT [options]`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from PIL import Image

from .edges import load_edge_mask
from .params import MosaicParams
from .pipeline import make_mosaic

app = typer.Typer(add_completion=False, help="Simulate a decorative tile mosaic from an image (Hausner 2001).")


def _parse_colour(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@app.command()
def main(
    input: Path = typer.Argument(..., exists=True, readable=True, help="Input image"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output image (default: <input>_mosaic.png)"),
    tiles: int = typer.Option(2000, "-n", "--tiles", help="Number of tiles"),
    tile_size: Optional[float] = typer.Option(None, "--tile-size", help="Tile side in px (at working resolution); overrides --tiles"),
    fill: float = typer.Option(0.8, help="Packing factor beta (tile area / region area)"),
    aspect: float = typer.Option(1.0, help="Tile elongation along edges (1 = square)"),
    max_size: int = typer.Option(1024, help="Working resolution (longest side)"),
    iters: int = typer.Option(25, help="Lloyd iterations with edge avoidance"),
    relax_iters: int = typer.Option(3, help="Final iterations without edges"),
    edges: Optional[Path] = typer.Option(None, "--edges", help="User edge mask image (white = edge) instead of Canny"),
    no_edges: bool = typer.Option(False, "--no-edges", help="Skip edges entirely (uniform orientation)"),
    edge_sigma: float = typer.Option(2.0, help="Canny blur sigma"),
    edge_low: float = typer.Option(0.1, help="Canny low threshold"),
    edge_high: float = typer.Option(0.3, help="Canny high threshold"),
    min_edge_len: float = typer.Option(2.0, help="Drop edge fragments shorter than this many tiles"),
    edge_width: float = typer.Option(0.3, help="Drawn edge thickness as a fraction of tile size"),
    detail_scale: float = typer.Option(1.0, help="Tile size near edges relative to base (<1 = smaller detail tiles)"),
    detail_radius: float = typer.Option(3.0, help="Distance in tiles over which tile size ramps back up"),
    color: str = typer.Option("mean", help="Tile colour: mean | point"),
    style: str = typer.Option("tiles", help="Output style: tiles | voronoi"),
    grout: str = typer.Option("#282828", help="Grout colour (hex)"),
    supersample: int = typer.Option(2, help="Anti-aliasing factor"),
    bevel: float = typer.Option(0.0, help="Darker tile outline strength 0..1"),
    jitter: float = typer.Option(0.0, help="Random per-tile brightness variation 0..1"),
    seed: int = typer.Option(0, help="Random seed"),
    debug_dir: Optional[Path] = typer.Option(None, "--debug-dir", help="Write edges/field/voronoi debug images here"),
):
    params = MosaicParams(
        tiles=tiles, tile_size=tile_size, fill=fill, aspect=aspect, max_size=max_size,
        iters=iters, relax_iters=relax_iters, edge_sigma=edge_sigma, edge_low=edge_low,
        edge_high=edge_high, min_edge_len=min_edge_len, edge_width=edge_width,
        detail_scale=detail_scale, detail_radius=detail_radius, color=color, style=style,
        grout=_parse_colour(grout), supersample=supersample, bevel=bevel, jitter=jitter, seed=seed,
    )
    image = Image.open(input)
    edge_mask = None
    if no_edges:
        import numpy as np
        edge_mask = np.zeros((image.height, image.width), bool)
    elif edges:
        edge_mask = load_edge_mask(str(edges), (image.height, image.width))

    t0 = time.time()

    def progress(stage, i, n):
        typer.echo(f"\r  {stage:<32} {i}/{n}", nl=False)

    result = make_mosaic(image, params, edge_mask, progress)
    typer.echo("")
    out = output or input.with_name(input.stem + "_mosaic.png")
    result.image.save(out)
    typer.echo(f"wrote {out}  ({len(result.state.sites)} tiles, {time.time() - t0:.1f}s)")
    if debug_dir:
        result.save_debug(debug_dir)
        typer.echo(f"debug images in {debug_dir}")
