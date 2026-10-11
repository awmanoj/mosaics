import numpy as np
from PIL import Image

from mosaic import MosaicParams, make_mosaic
from mosaic.cvd import assign_regions, square_metric


def _disc_image(size=160):
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size, 3), 230, np.uint8)
    img[(xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size / 3) ** 2] = (200, 40, 40)
    return Image.fromarray(img)


def test_square_metric_level_sets_are_axis_aligned_squares():
    # the unit level set is a square with sides along the (rotated) axes
    assert np.isclose(square_metric(1.0, 0.0, 0.0, 1.0, 1.0), 1.0)
    assert np.isclose(square_metric(1.0, 1.0, 0.0, 1.0, 1.0), 1.0)      # corner of the square
    assert np.isclose(square_metric(1.0, 1.0, np.pi / 4, 1.0, 1.0), np.sqrt(2))  # rotated frame
    assert np.isclose(square_metric(0.0, 2.0, 0.0, 1.0, 2.0), 1.0)      # aspect stretches v
    assert np.isclose(square_metric(2.0, 0.0, 0.0, 2.0, 1.0), 1.0)      # size lowers the slope


def test_assign_regions_covers_every_pixel_and_blocks_edges():
    sites = np.array([[10.0, 10.0], [50.0, 50.0], [10.0, 50.0]])
    angles = np.zeros(3, np.float32)
    sizes = np.ones(3, np.float32)
    blocked = np.zeros((64, 64), bool)
    blocked[30, :] = True
    labels = assign_regions(sites, angles, sizes, 1.0, (64, 64), 8, blocked)
    assert labels.shape == (64, 64)
    assert (labels[30] == -1).all()
    assert set(np.unique(labels[labels >= 0])) == {0, 1, 2}


def test_make_mosaic_end_to_end():
    img = _disc_image()
    params = MosaicParams(tiles=120, iters=8, relax_iters=2, max_size=160, supersample=1)
    res = make_mosaic(img, params)
    assert res.image.size == img.size
    assert len(res.state.sites) == 120
    assert res.edges.any()  # the disc outline is detected
    # tiles should be red inside and light outside
    reds = (res.colours[:, 0] > 0.6) & (res.colours[:, 1] < 0.4)
    assert reds.any() and (~reds).any()


def test_voronoi_style_and_point_colour():
    img = _disc_image(96)
    params = MosaicParams(tiles=40, iters=3, relax_iters=1, max_size=96, style="voronoi", color="point")
    res = make_mosaic(img, params)
    assert res.image.size == img.size


def test_exif_orientation_is_applied():
    img = _disc_image(96).convert("RGB")
    wide = img.resize((120, 60))
    exif = wide.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    import io
    buf = io.BytesIO()
    wide.save(buf, format="JPEG", exif=exif.tobytes())
    loaded = Image.open(io.BytesIO(buf.getvalue()))
    res = make_mosaic(loaded, MosaicParams(tiles=30, iters=2, relax_iters=1, max_size=120))
    assert res.image.size == (60, 120)


def test_out_max_size_keeps_whole_image():
    import numpy as np
    img = _disc_image(200)
    base = MosaicParams(tiles=80, iters=3, relax_iters=1, max_size=200, supersample=1)
    full = make_mosaic(img, base).image
    small = make_mosaic(img, MosaicParams(**{**base.__dict__, "out_max_size": 100})).image
    assert small.size == (100, 100)
    a = np.asarray(full.resize((100, 100), Image.LANCZOS), dtype=float)
    b = np.asarray(small, dtype=float)
    assert np.abs(a - b).mean() < 25  # same picture, only resampled differently
