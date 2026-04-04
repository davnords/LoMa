import numpy as np
from PIL import Image
import warnings


def numpy_to_pil(x: np.ndarray):
    assert x.dtype in [np.float32, np.uint8]
    if x.dtype == np.float32:
        assert x.min() >= 0.0 and x.max() <= 1.0
        x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x, autoscale=False):
    x = x.clone().detach().cpu().numpy()
    if autoscale:
        if x.max() == x.min():
            warnings.warn("x max == x min, cant autoscale")
        else:
            x = (x - x.min()) / (x.max() - x.min())

    if len(x.shape) == 3:
        x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def check_not_i16(pil_img: Image.Image):
    if pil_img.mode == "I;16":
        raise NotImplementedError("Can't handle 16 bit images")
