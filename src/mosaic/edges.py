"""Edge features. The paper uses user-drawn curves; here we auto-detect with Canny by default."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import remove_small_objects


def detect_edges(rgb: np.ndarray, sigma: float, low: float, high: float, min_len: float) -> np.ndarray:
    """Return a boolean (H, W) mask of thin edge curves from an RGB float image in [0, 1]."""
    gray = rgb2gray(rgb)
    mask = canny(gray, sigma=sigma, low_threshold=low, high_threshold=high)
    if min_len > 1:
        # edges are 1px thick, so a component's pixel count approximates its length
        try:
            mask = remove_small_objects(mask, max_size=int(min_len), connectivity=2)
        except TypeError:  # scikit-image < 0.26
            mask = remove_small_objects(mask, min_size=int(min_len), connectivity=2)
    return mask


def load_edge_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    """Load a user-drawn mask (white = edge) and resize it to (H, W)."""
    h, w = shape
    img = Image.open(path).convert("L").resize((w, h), Image.NEAREST)
    return np.asarray(img) > 127
