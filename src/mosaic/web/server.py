"""Minimal web UI: one page, upload an image, get the mosaic back. `uv run mosaic-web`."""

from __future__ import annotations

import asyncio
import html
import io
import os
import re
import secrets
import string
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ..params import MosaicParams
from ..pipeline import make_mosaic

app = FastAPI(title="mosaic")
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024
_lock = asyncio.Semaphore(2)  # the engine is CPU bound; don't let many jobs run at once
_ALPHABET = string.ascii_lowercase + string.digits
app.mount("/examples", StaticFiles(directory=STATIC / "examples"), name="examples")


def _code(n: int = 6) -> str:
    """Short lowercase alphanumeric id used to name each result, e.g. 'k7f3q2'."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(n))


def _hex(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


_GA_SNIPPET = """<script async src="https://www.googletagmanager.com/gtag/js?id={id}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{id}');
</script>"""


def _ga_tag() -> str:
    """Google Analytics snippet for GA_MEASUREMENT_ID (e.g. G-XXXXXXXXXX), or '' when unset."""
    gid = os.environ.get("GA_MEASUREMENT_ID", "").strip()
    if not gid or not re.fullmatch(r"[A-Za-z0-9-]+", gid):
        return ""
    return _GA_SNIPPET.format(id=html.escape(gid))


@app.get("/")
async def index() -> HTMLResponse:
    page = (STATIC / "index.html").read_text().replace("<!--GA-->", _ga_tag(), 1)
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


@app.post("/mosaic")
async def mosaic(
    image: UploadFile = File(...),
    tiles: int = Form(2000),
    iters: int = Form(30),
    fill: float = Form(0.8),
    aspect: float = Form(1.0),
    detail_scale: float = Form(1.0),
    edge_sigma: float = Form(2.0),
    edge_width: float = Form(0.3),
    color: str = Form("mean"),
    style: str = Form("tiles"),
    grout: str = Form("#282828"),
    bevel: float = Form(0.0),
    jitter: float = Form(0.0),
    max_size: int = Form(800),
    seed: int = Form(0),
    edges: Optional[UploadFile] = File(None),
):
    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "not an image")

    edge_mask = None
    if edges is not None and edges.filename:
        import numpy as np
        m = Image.open(io.BytesIO(await edges.read())).convert("L").resize(img.size, Image.NEAREST)
        edge_mask = np.asarray(m) > 127

    params = MosaicParams(
        tiles=max(10, min(tiles, 20000)),
        iters=max(1, min(iters, 150)),
        relax_iters=3,
        fill=max(0.2, min(fill, 1.2)),
        aspect=max(0.25, min(aspect, 4.0)),
        detail_scale=max(0.2, min(detail_scale, 1.0)),
        edge_sigma=max(0.5, min(edge_sigma, 8.0)),
        edge_width=max(0.0, min(edge_width, 1.5)),
        color=color if color in ("mean", "point") else "mean",
        style=style if style in ("tiles", "voronoi") else "tiles",
        grout=_hex(grout),
        bevel=max(0.0, min(bevel, 1.0)),
        jitter=max(0.0, min(jitter, 1.0)),
        max_size=max(200, min(max_size, 1600)),
        out_max_size=2000,  # keeps the PNG and the browser-side paint manageable for phone photos
        seed=seed,
    )

    async with _lock:
        result = await asyncio.to_thread(make_mosaic, img, params, edge_mask)

    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    name = f"mosaic-{_code()}.png"
    return Response(
        buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Tiles": str(len(result.state.sites)),
            "X-Name": name,
            "Content-Disposition": f'inline; filename="{name}"',
        },
    )


def main() -> None:
    """Start the server. Env: MOSAIC_HOST (127.0.0.1), MOSAIC_PORT (8765)."""
    import os
    import uvicorn

    host = os.environ.get("MOSAIC_HOST", "127.0.0.1")
    port = int(os.environ.get("MOSAIC_PORT", "8765"))
    print(f"mosaic web UI: http://{host}:{port}/")
    uvicorn.run("mosaic.web.server:app", host=host, port=port, reload=False)
