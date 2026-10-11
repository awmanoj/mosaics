# mosaic — Simulating Decorative Mosaics

A CPU implementation of Alejo Hausner's *Simulating Decorative Mosaics* (SIGGRAPH 2001,
[doi:10.1145/383259.383327](https://doi.org/10.1145/383259.383327)), packaged as a Python
library with a command-line front end. A small web front end is planned on
top of the same `make_mosaic()` API.

## The algorithm (what the paper does, and what this code does)

1. **Edge features** (sec 4). The paper uses curves drawn by the artist. Here edges are detected
   with Canny by default (`--edge-sigma/--edge-low/--edge-high`), short fragments are dropped, or
   you can pass your own mask with `--edges mask.png` (white = edge).
2. **Direction field φ** (sec 4). The Euclidean distance transform of the edge mask is the
   "mountain range" of the paper; its gradient gives φ, the direction perpendicular to the nearest
   edge. Because square tiles have 90° symmetry, φ is smoothed in (cos 4φ, sin 4φ) space.
3. **Centroidal Voronoi diagram with a rotated square metric** (sec 3). Each site owns the pixels
   where its square pyramid, rotated to the local φ, is lowest. The paper rasterises the pyramids on
   a GPU z-buffer; we use a k-d tree to propose the nearest `--neighbours` sites per pixel and pick
   the lowest pyramid among them, fully vectorised in numpy. Lloyd's method moves every site to the
   centroid of its region and repeats (`--iters`).
4. **Edge avoidance** (sec 6). The edges are drawn as thick lines (`--edge-width`, as a fraction of
   tile size) that belong to no site, so straddling tiles get pushed off the edges. A few final
   iterations without the lines (`--relax-iters`) close the gaps.
5. **Tiles** (sec 5). Tile side `d = sqrt(β·h·w/n)` with β = `--fill` (0.8 in the paper). Colour is
   the mean over the tile's Voronoi region or a point sample (`--color`). Tile size can shrink near
   edges (`--detail-scale`, `--detail-radius`; initial sites are rejection-sampled with
   probability (s_min/s)² as in the paper) and tiles can be elongated along edges (`--aspect`).
   `--style voronoi` outputs the filled regions instead (Haeberli-style, paper fig 12b).

**One deviation worth knowing.** The paper writes the pyramid as the Manhattan metric
`|u| + |v|`. The level sets of that pyramid are diamonds, and Lloyd's method under it settles into
square cells whose sides lie at 45° to the metric axes (the diamond cell has lower L1 energy).
Drawing tiles aligned with φ then puts them at 45° to their cells. The pyramid whose base is a
square aligned with φ is `max(|u|, |v|)` (Chebyshev), which is the same pyramid rotated 45°. We use
that form; see the docstring in `src/mosaic/cvd.py`.

## Install and run

```bash
uv sync
uv run mosaic examples/yinyang.png -o out/yinyang.png -n 1000
uv run mosaic photo.jpg -n 3000 --iters 40 --bevel 0.3 --detail-scale 0.6
uv run mosaic --help
```

Useful flags:

| flag | meaning |
|---|---|
| `-n/--tiles`, `--tile-size` | number of tiles, or tile side in px at working resolution |
| `--fill` | β, tile area / cell area (0.8) |
| `--iters`, `--relax-iters` | Lloyd iterations with and without the edge lines (25 + 3) |
| `--max-size` | working resolution; output keeps the input size |
| `--edges`, `--no-edges` | user edge mask / no edges at all |
| `--edge-width` | thickness of the avoidance lines in tile units |
| `--detail-scale`, `--detail-radius` | smaller tiles near edges |
| `--aspect` | elongate tiles along edges (hair, strokes) |
| `--color mean\|point`, `--style tiles\|voronoi`, `--grout #rrggbb`, `--bevel`, `--jitter` | appearance |
| `--debug-dir` | writes `edges.png`, `field.png`, `voronoi.png`, `regions.png` |

Regenerate the sample images with `uv run python examples/make_samples.py`.

## Library use

```python
from PIL import Image
from mosaic import MosaicParams, make_mosaic

result = make_mosaic(Image.open("photo.jpg"), MosaicParams(tiles=2000))
result.image.save("photo_mosaic.png")
# result.state.sites / .angles, result.sides, result.colours, result.phi, result.edges
```

`make_mosaic(image, params, edge_mask=None, progress=None)` is the only entry point the web app
will need.

## Layout

```
src/mosaic/
  params.py    MosaicParams dataclass (all knobs, paper defaults)
  edges.py     Canny edge detection / user edge masks
  field.py     distance transform -> direction field, size-factor map
  cvd.py       rotated square metric, region assignment, Lloyd's method
  render.py    tile and Voronoi renderers, debug images
  pipeline.py  make_mosaic() end to end
  cli.py       typer CLI
tests/         pytest smoke tests
examples/      sample images + generator
```

## Tips

- Photos with lots of texture produce many Canny edges and a fragmented field. Raise
  `--edge-sigma` (3–4) or `--edge-high`, or supply a hand-drawn `--edges` mask for the few curves
  that matter, which is what the paper does.
- More iterations give more regular grids; 40 is a good default for 2–3k tiles (a few seconds).

## Web UI

```bash
uv run mosaic-web          # http://127.0.0.1:8765/  (MOSAIC_PORT / MOSAIC_HOST to change)
```

One page: drop, paste or pick an image, move the sliders, press **Make mosaic**, download the PNG.
The page posts a multipart form to `POST /mosaic` (fields mirror the CLI flags, plus an optional
`edges` mask upload) and gets a PNG back; the `X-Tiles` header carries the tile count. The server
runs the engine in a worker thread and allows two jobs at a time. Code: `src/mosaic/web/`.

## Docker

```bash
docker build -t mosaic .
docker run --rm -p 8765:8765 mosaic        # then open http://localhost:8765/
# or
docker compose up -d
```

The container listens on `0.0.0.0:8765`; put a reverse proxy with TLS in front of it on a public
server. Set `MOSAIC_PORT` to change the port inside the container.

Environment variables:

| variable | effect |
|---|---|
| `MOSAIC_HOST`, `MOSAIC_PORT` | bind address (defaults `127.0.0.1:8765`; the image sets `0.0.0.0`) |
| `GA_MEASUREMENT_ID` | when set (e.g. `G-XXXXXXXXXX`), the Google Analytics gtag snippet is injected into the page; unset means no tracking |
# mosaics
