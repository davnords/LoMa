from dataclasses import field
from dataclasses import dataclass
import math
import random
from typing import Literal
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
import torchvision.transforms.functional as tvf

from ..distrib import get_rank
from ..logging import logger
from ..io import check_not_i16
from ..geometry import from_homogeneous, get_normalized_grid, to_homogeneous

# From Patch2Pix https://github.com/GrumpyZhou/patch2pix
def get_depth_tuple_transform_ops(
    *,
    resize: tuple[int | None, int | None],
    rtol: float | None,
    atol: float | None,
    mode: InterpolationMode,
):
    if resize[0] is None or resize[1] is None:
        ops = []
    else:
        ops = [
            TupleResize(
                resize,
                # FIXME: don't push without making configurable
                mode=mode,
                nearest_fallback=True,
                # mode=InterpolationMode.BILINEAR,
                # nearest_fallback=True,
                antialias=False,
                atol=atol,
                rtol=rtol,
            )
        ]
    return TupleCompose(ops)


def get_flow_tuple_transform_ops(
    *,
    resize: tuple[int | None, int | None],
    atol: float | None,
    rtol: float | None,
    mode: InterpolationMode,
):
    if resize[0] is None or resize[1] is None:
        ops = []
    else:
        ops = [
            TupleResize(
                resize,
                mode=mode,
                antialias=False,
                nearest_fallback=True,
                atol=atol,
                rtol=rtol,
            )
        ]
    return TupleCompose(ops)


def get_mask_tuple_transform_ops(
    *, resize: tuple[int | None, int | None], mode: InterpolationMode
):
    if resize[0] is None or resize[1] is None:
        ops = []
    else:
        ops = [TupleResize(resize, mode=mode, antialias=False)]
    return TupleCompose(ops)


def get_tuple_transform_ops(*, resize: tuple[int | None, int | None]):
    if resize[0] is None or resize[1] is None:
        ops = [TupleToTensorScaled()]
    else:
        ops = [
            TupleResize(resize, mode=InterpolationMode.BICUBIC, antialias=True),
            TupleToTensorScaled(),
        ]
    return TupleCompose(ops)


class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class TupleResize:
    def __init__(
        self,
        size,
        antialias: bool | None,
        mode: InterpolationMode,
        nearest_fallback: bool = False,
        atol: float | None = None,
        rtol: float | None = None,
    ):
        self.size = size
        self.resize = transforms.Resize(size, mode, antialias=antialias)  # pyright: ignore[reportArgumentType]
        if nearest_fallback:
            self.nearest_resize = transforms.Resize(
                size, InterpolationMode.NEAREST_EXACT, antialias=False
            )  # pyright: ignore[reportArgumentType]
        else:
            self.nearest_resize = None
        self.atol = atol if atol is not None else 0.0
        self.rtol = rtol if rtol is not None else 0.0

    def __call__(self, im_tuple):
        res = [self.resize(im) for im in im_tuple]
        if self.nearest_resize is not None:
            res_nearest = [self.nearest_resize(im) for im in im_tuple]
            # check for discrepencies, and fallback to nearest if necessary
            for im, im_nearest in zip(res, res_nearest, strict=True):
                # we can have relatively relaxed tolerances, since we mainly want to remove large outliers
                is_close = torch.isclose(
                    im,
                    im_nearest,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                # override with nearest
                im[~is_close] = im_nearest[~is_close]
        return res

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)


class TupleCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def crop(img: Image.Image, x: int, y: int, crop_size: int):
    width, height = img.size
    if width < crop_size or height < crop_size:
        raise ValueError(f"Image dimensions must be at least {crop_size}x{crop_size}")
    cropped_img = img.crop((x, y, x + crop_size, y + crop_size))
    return cropped_img


def random_crop(img: Image.Image, crop_size: int):
    width, height = img.size

    if width < crop_size or height < crop_size:
        raise ValueError(f"Image dimensions must be at least {crop_size}x{crop_size}")

    max_x = width - crop_size
    max_y = height - crop_size

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    cropped_img = img.crop((x, y, x + crop_size, y + crop_size))
    return cropped_img, (x, y)


def apply_random_hue(pil_img: Image.Image) -> Image.Image:
    rgb_array = np.array(pil_img)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    # https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
    hsv[..., 0] = (
        (hsv[..., 0].astype(np.int32) + np.random.randint(-15, 15)) % 180
    ).astype(np.uint8)
    rgb_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb_array)


def apply_random_value(pil_img: Image.Image) -> Image.Image:
    rgb_array = np.array(pil_img)
    hsv: np.ndarray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    bigger_than_zero = hsv[..., 2] > 0
    if not np.any(bigger_than_zero):
        logger.info(
            "No hsv value is greater than 0, returning original image, don't worry."
        )
        return pil_img
    median_value = np.median(hsv[..., 2][bigger_than_zero])
    if np.isnan(median_value):
        logger.warning("Median hsv value is NaN, probably shouldn't happen, worry.")
        return pil_img
    max_factor = 1.5
    min_brightness = max(min(25, median_value), median_value / max_factor)
    max_brightness = min(255, max_factor * median_value)
    a_max = max_brightness / median_value
    a_min = min_brightness / median_value
    brightness_factor = np.random.rand(1) * (a_max - a_min) + a_min
    hsv[..., 2] = (
        (hsv[..., 2].astype(np.float32) * brightness_factor)
        .clip(0, 255)
        .astype(np.uint8)
    )
    rgb_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb_array)


def apply_random_blur(pil_img: Image.Image) -> Image.Image:
    rgb_array = np.array(pil_img)
    rgb_array = cv2.GaussianBlur(rgb_array, (5, 5), 0)
    return Image.fromarray(rgb_array)


def scale_intrinsic(
    *, K: torch.Tensor, W_from: int, H_from: int, W_to: int, H_to: int
) -> torch.Tensor:
    sx, sy = W_to / W_from, H_to / H_from
    sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    return sK @ K


def rand_shake(*things, shake_t: int):
    t = np.random.choice(range(-shake_t, shake_t + 1), size=(2))
    return [
        tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
        for thing in things
    ], torch.from_numpy(t)


def rot_360_deg(
    *,
    im: torch.Tensor,
    depth: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    angle: float | int,
    flow: torch.Tensor | None,
):
    device = im.device
    one, C, H, W = im.shape
    assert one == 1
    radians = angle * math.pi / 180
    rot_mat = torch.tensor(
        [
            [math.cos(radians), math.sin(radians), 0],
            [-math.sin(radians), math.cos(radians), 0],
            [0, 0, 1.0],
        ]
    ).to(device)
    im = tvf.rotate(im, angle, expand=True)
    if depth is not None:
        depth = tvf.rotate(depth, angle, expand=True)
    if flow is not None:
        flow = tvf.rotate(flow.permute(0, 3, 1, 2), angle, expand=True).permute(
            0, 2, 3, 1
        )
    if intrinsics is not None:
        t_mat = torch.tensor([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1.0]]).to(device)
        neg_t_mat = torch.tensor([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1.0]]).to(
            device
        )
        transform = t_mat @ rot_mat @ neg_t_mat
        intrinsics = transform @ intrinsics
    return im, depth, intrinsics, flow, rot_mat


def horizontal_flip(
    *,
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    depth_A: torch.Tensor | None,
    depth_B: torch.Tensor | None,
    flow_AB: torch.Tensor | None,
    flow_BA: torch.Tensor | None,
    correspondences_AB: torch.Tensor | None,
    K_A: torch.Tensor | None,
    K_B: torch.Tensor | None,
    mask_AB: torch.Tensor | None,
    mask_BA: torch.Tensor | None,
    width: int,
):
    device = img_A.device
    img_A = img_A.flip(-1)
    img_B = img_B.flip(-1)
    if depth_A is not None and depth_B is not None:
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1)
    if flow_AB is not None and flow_BA is not None:
        flow_AB[..., 0] *= -1
        flow_BA[..., 0] *= -1
        flow_AB = flow_AB.flip(-2)
        flow_BA = flow_BA.flip(-2)
    if mask_AB is not None and mask_BA is not None:
        mask_AB = mask_AB.flip(-1)
        mask_BA = mask_BA.flip(-1)
    if K_A is not None and K_B is not None:
        flip_mat = torch.tensor([[-1, 0, width], [0, 1, 0], [0, 0, 1.0]]).to(device)
        K_A = flip_mat @ K_A
        K_B = flip_mat @ K_B
    if correspondences_AB is not None:
        correspondences_AB[..., 0] *= -1
        correspondences_AB[..., 2] *= -1
    return (
        img_A,
        img_B,
        depth_A,
        depth_B,
        flow_AB,
        flow_BA,
        correspondences_AB,
        mask_AB,
        mask_BA,
        K_A,
        K_B,
    )


def luminance_negation(pil_img: Image.Image) -> Image.Image:
    rgb_array = np.array(pil_img)
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    # Negate L channel
    lab[:, :, 0] = 255 - lab[:, :, 0]
    bgr_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb_result = cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_result)


class Transform:
    @dataclass(frozen=True)
    class Cfg:
        heights: list[int | None] | None = field(
            default_factory=lambda: [
                512,
                592,
                624,
                688,
                512,
                448,
                416,
                384,
            ]
        )
        widths: list[int | None] | None = field(
            default_factory=lambda: [
                512,
                448,
                416,
                384,
                512,
                592,
                624,
                688,
            ]
        )
        shake_t: int = 0
        random_hue: bool = True  # True
        random_value: bool = True  # True
        use_horizontal_flip: bool = True  # True
        grayscale_prob: float = 0.1  # 0.1
        depth_rtol: float | None = 1e-2
        depth_atol: float | None = 1e-2
        flow_atol: float | None = 1e-2
        flow_rtol: float | None = 1e-2
        depth_interpolation_mode: Literal["nearest", "bilinear"] = "bilinear"
        flow_interpolation_mode: Literal["nearest", "bilinear"] = "bilinear"
        mask_interpolation_mode: Literal["nearest", "bilinear"] = "nearest"
        rot360: bool = False
        joint_rot: bool = False

    @dataclass(frozen=True)
    class BenchmarkCfg(Cfg):
        heights: list[int | None] | None = field(
            default_factory=lambda: [
                # 420,
                640,
            ]
        )
        widths: list[int | None] | None = field(
            default_factory=lambda: [
                # 560,
                640,
            ]
        )
        random_hue: bool = False  # True
        random_value: bool = False  # True
        use_horizontal_flip: bool = False  # True
        grayscale_prob: float = 0.0  # 0.1
        rot360: bool = False

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        str_to_interpolation_mode = {
            "nearest": InterpolationMode.NEAREST_EXACT,
            "bilinear": InterpolationMode.BILINEAR,
        }
        depth_interpolation_mode = str_to_interpolation_mode[
            cfg.depth_interpolation_mode
        ]
        flow_interpolation_mode = str_to_interpolation_mode[cfg.flow_interpolation_mode]
        mask_interpolation_mode = str_to_interpolation_mode[cfg.mask_interpolation_mode]
        rank = get_rank()
        if cfg.heights is not None:
            self.height = cfg.heights[rank % len(cfg.heights)]
        else:
            self.height = None
        if cfg.widths is not None:
            self.width = cfg.widths[rank % len(cfg.widths)]
        else:
            self.width = None
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(self.height, self.width),
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
            resize=(self.height, self.width),
            rtol=self.cfg.depth_rtol,
            atol=self.cfg.depth_atol,
            mode=depth_interpolation_mode,
        )
        self.flow_transform_ops = get_flow_tuple_transform_ops(
            resize=(self.height, self.width),
            atol=self.cfg.flow_atol,
            rtol=self.cfg.flow_rtol,
            mode=flow_interpolation_mode,
        )
        self.mask_transform_ops = get_mask_tuple_transform_ops(
            resize=(self.height, self.width),
            mode=mask_interpolation_mode,
        )
        self.grid = None

    def __call__(
        self,
        *,
        pil_img_A: Image.Image,
        pil_img_B: Image.Image,
        depth_A: torch.Tensor | None = None,
        depth_B: torch.Tensor | None = None,
        flow_AB: torch.Tensor | None = None,
        flow_BA: torch.Tensor | None = None,
        correspondences_AB: torch.Tensor | None = None,
        mask_AB: torch.Tensor | None = None,
        mask_BA: torch.Tensor | None = None,
        K_A: torch.Tensor | None = None,
        K_B: torch.Tensor | None = None,
    ):
        ret = {}
        W_A, H_A = pil_img_A.width, pil_img_A.height
        W_B, H_B = pil_img_B.width, pil_img_B.height

        check_not_i16(pil_img_A)
        check_not_i16(pil_img_B)
        pil_img_A, pil_img_B = pil_img_A.convert("RGB"), pil_img_B.convert("RGB")

        if self.cfg.random_hue:
            pil_img_A = apply_random_hue(pil_img_A)
            pil_img_B = apply_random_hue(pil_img_B)

        if self.cfg.random_value:
            pil_img_A = apply_random_value(pil_img_A)
            pil_img_B = apply_random_value(pil_img_B)

        if np.random.rand() < self.cfg.grayscale_prob:
            pil_img_A = pil_img_A.convert("L").convert("RGB")
        if np.random.rand() < self.cfg.grayscale_prob:
            pil_img_B = pil_img_B.convert("L").convert("RGB")

        if (
            self.height is not None
            and self.width is not None
            and K_A is not None
            and K_B is not None
        ):
            K_A = scale_intrinsic(
                K=K_A,
                W_from=W_A,
                H_from=H_A,
                W_to=self.width,
                H_to=self.height,
            )
            K_B = scale_intrinsic(
                K=K_B,
                W_from=W_B,
                H_from=H_B,
                W_to=self.width,
                H_to=self.height,
            )

        img_A, img_B = self.im_transform_ops((pil_img_A, pil_img_B))
        if depth_A is not None and depth_B is not None:
            depth_A, depth_B = self.depth_transform_ops(
                (depth_A[None, None], depth_B[None, None])
            )
        if mask_AB is not None and mask_BA is not None:
            mask_AB, mask_BA = self.mask_transform_ops(
                (mask_AB[None, None], mask_BA[None, None])
            )

        if flow_AB is not None and flow_BA is not None:
            flow_AB, flow_BA = self.flow_transform_ops(
                (flow_AB.permute(2, 0, 1)[None], flow_BA.permute(2, 0, 1)[None])
            )
            flow_AB = flow_AB.permute(0, 2, 3, 1)
            flow_BA = flow_BA.permute(0, 2, 3, 1)

        img_A, img_B = img_A[None], img_B[None]

        if self.cfg.rot360:
            angle_A = np.random.choice([-90, 0, 90, 180])
            angle_B = angle_A if self.cfg.joint_rot else np.random.choice([-90, 0, 90, 180])
            angle_A, angle_B = int(angle_A), int(angle_B)
            self.grid = (
                get_normalized_grid(
                    1,
                    H=self.height if self.height is not None else H_A,
                    W=self.width if self.width is not None else W_A,
                    overload_device=img_A.device,
                )
                if self.grid is None
                else self.grid
            )
            warp_AB = None
            warp_BA = None
            if flow_AB is not None and flow_BA is not None:
                warp_AB = self.grid + flow_AB
                warp_BA = self.grid + flow_BA
            # first rotate pixels and compute transformations
            img_A, depth_A, K_A, warp_AB, rot_A = rot_360_deg(
                im=img_A, depth=depth_A, intrinsics=K_A, angle=angle_A, flow=warp_AB
            )

            img_B, depth_B, K_B, warp_BA, rot_B = rot_360_deg(
                im=img_B, depth=depth_B, intrinsics=K_B, angle=angle_B, flow=warp_BA
            )
            if warp_AB is not None and warp_BA is not None:
                # then rotate warp which refers to old coordinate system of other to new one
                warp_AB = to_homogeneous(warp_AB) @ rot_B.mT
                warp_BA = to_homogeneous(warp_BA) @ rot_A.mT
                flow_AB = from_homogeneous(warp_AB) - self.grid
                flow_BA = from_homogeneous(warp_BA) - self.grid
            if correspondences_AB is not None:
                correspondences_AB[..., :2] = (
                    correspondences_AB[..., :2] @ rot_A[..., :2, :2].mT
                )
                correspondences_AB[..., 2:] = (
                    correspondences_AB[..., 2:] @ rot_B[..., :2, :2].mT
                )
        if (
            self.cfg.shake_t > 0
            and depth_A is not None
            and depth_B is not None
            and K_A is not None
            and K_B is not None
        ):
            [img_A, depth_A], t_A = rand_shake(img_A, depth_A, shake_t=self.cfg.shake_t)
            [img_B, depth_B], t_B = rand_shake(img_B, depth_B, shake_t=self.cfg.shake_t)

            K_A[:2, 2] += t_A
            K_B[:2, 2] += t_B

        if self.cfg.use_horizontal_flip and np.random.rand() > 0.5:
            horz_flip_width = W_A if self.width is None else self.width
            (
                img_A,
                img_B,
                depth_A,
                depth_B,
                flow_AB,
                flow_BA,
                correspondences_AB,
                mask_AB,
                mask_BA,
                K_A,
                K_B,
            ) = horizontal_flip(
                img_A=img_A,
                img_B=img_B,
                depth_A=depth_A,
                depth_B=depth_B,
                flow_AB=flow_AB,
                flow_BA=flow_BA,
                correspondences_AB=correspondences_AB,
                K_A=K_A,
                K_B=K_B,
                mask_AB=mask_AB,
                mask_BA=mask_BA,
                width=horz_flip_width,
            )
        img_A, img_B = img_A[0], img_B[0]
        ret["img_A"] = img_A
        ret["img_B"] = img_B

        # prepare batch
        H, W = img_A.shape[1:3]
        if depth_A is not None and depth_B is not None:
            ret["depth_A"] = depth_A[0, 0, ..., None]
            ret["depth_B"] = depth_B[0, 0, ..., None]
        else:
            ret["depth_A"] = torch.zeros((H, W, 1))
            ret["depth_B"] = torch.zeros((H, W, 1))
        if mask_AB is not None and mask_BA is not None:
            ret["mask_AB"] = mask_AB[0, 0, ..., None]
            ret["mask_BA"] = mask_BA[0, 0, ..., None]
        else:
            ret["mask_AB"] = torch.ones((H, W, 1), dtype=torch.bool)
            ret["mask_BA"] = torch.ones((H, W, 1), dtype=torch.bool)
        if flow_AB is not None and flow_BA is not None:
            ret["flow_AB"] = flow_AB[0]
            ret["flow_BA"] = flow_BA[0]
        else:
            ret["flow_AB"] = torch.zeros((H, W, 2))
            ret["flow_BA"] = torch.zeros((H, W, 2))

        if K_A is not None:
            ret["K_A"] = K_A
        if K_B is not None:
            ret["K_B"] = K_B
        ret["img_A"] = img_A
        ret["img_B"] = img_B
        if correspondences_AB is not None:
            ret["correspondences_AB"] = correspondences_AB
        return ret
