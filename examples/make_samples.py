"""Generate synthetic test images (a yin-yang like the paper's figure 8)."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def yin_yang(size=700):
    img = Image.new("RGB", (size, size), (200, 30, 30))
    d = ImageDraw.Draw(img)
    c, r = size / 2, size * 0.42
    d.ellipse([c - r, c - r, c + r, c + r], fill=(255, 255, 255))
    d.pieslice([c - r, c - r, c + r, c + r], 90, 270, fill=(0, 0, 0))
    for sign, col in ((-1, (255, 255, 255)), (1, (0, 0, 0))):
        cy = c + sign * r / 2
        d.ellipse([c - r / 2, cy - r / 2, c + r / 2, cy + r / 2], fill=col)
        d.ellipse([c - r / 7, cy - r / 7, c + r / 7, cy + r / 7], fill=(255, 255, 255) if col == (0, 0, 0) else (0, 0, 0))
    return img


if __name__ == "__main__":
    out = Path(__file__).parent
    yin_yang().save(out / "yinyang.png")
    print("wrote", out / "yinyang.png")
