"""Detections must be flip-equivariant at any image size, not only multiples of 4.

`to_pixel_coords` maps a score-map index i to i + 0.5 on a [0, W] axis, so a horizontal flip
maps x
to W - x exactly: the feature at index i of the mirrored image is index W - 1 - i of the
original,
and W - (i + 0.5) == (W - 1 - i) + 0.5. Detecting on an image and on its mirror and mapping back
must
therefore agree to the subpixel, whatever the width.

Before the padding fix in forward_impl this failed for every width not divisible by 4, reaching
about
half a pixel: the encoder's floor-halvings gave a non-integral F.interpolate ratio, which moved
the
score map off the pixel grid. `DaD.load_image` floors both dimensions to a multiple of 8, which
is also a multiple of 4, so the documented path hid this; a caller passing its own tensor did
not.
"""

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from scipy.stats import trim_mean

from loma.detector.dad import DaD

NUM_KEYPOINTS = 2048
MATCH_RADIUS = 2.0
TOLERANCE = 0.05


@pytest.fixture(scope="module")
def detector():
    return DaD(DaD.Cfg(compile=False)).eval()


def _pixel_keypoints(detector, image):
    tensor = torch.from_numpy(image.transpose(2, 0, 1)[None] / 255.0).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.inference_mode():
        out = detector.detect({"image": tensor}, num_keypoints=NUM_KEYPOINTS)
    kpts = out["keypoints"]
    if kpts.ndim == 3:
        kpts = kpts[0]
    coords = detector.to_pixel_coords(kpts, image.shape[0], image.shape[1])
    return coords.detach().cpu().numpy().astype(np.float64)


def _flip_offset(detector, image, axis):
    """Trimmed-mean residual after mapping mirrored detections back. Exactly 0 when
equivariant."""
    height, width = image.shape[0], image.shape[1]
    original = _pixel_keypoints(detector, image)
    if axis == 0:
        mirrored = _pixel_keypoints(detector, np.ascontiguousarray(image[:, ::-1]))
        mirrored[:, 0] = width - mirrored[:, 0]
    else:
        mirrored = _pixel_keypoints(detector, np.ascontiguousarray(image[::-1, :]))
        mirrored[:, 1] = height - mirrored[:, 1]
    distance, index = cKDTree(original).query(mirrored, k=1)
    near = distance < MATCH_RADIUS
    assert near.sum() > 50, "too few correspondences to measure"
    return float(trim_mean((mirrored[near] - original[index[near]])[:, axis], 0.2))


# One size per residue class, so a size-dependent bias cannot hide in an untested residue.
@pytest.mark.parametrize("size", [497, 498, 499, 500, 501, 502, 503, 504])
@pytest.mark.parametrize("axis", [0, 1])
def test_flip_equivariance_at_any_size(detector, size, axis):
    from skimage import data

    base = np.asarray(data.astronaut())[..., :3].astype(np.uint8)
    shape = (512, size) if axis == 0 else (size, 512)
    image = np.ascontiguousarray(base[: shape[0], : shape[1]])
    offset = _flip_offset(detector, image, axis)
    assert abs(offset) < TOLERANCE, (
        f"{'width' if axis == 0 else 'height'}={size} "
        f"({'W' if axis == 0 else 'H'}%4={size % 4}): offset {offset:+.3f} px, expected 0"
    )
