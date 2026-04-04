from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from loma.io import tensor_to_pil
from loma.geometry import kde, get_normalized_grid
from loma.types import Detector


def extract_patches_from_inds(x: torch.Tensor, inds: torch.Tensor, patch_size: int):
    B, H, W = x.shape
    B, N = inds.shape
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])  # B x K_H * K_W x H * W
    patches = torch.gather(
        unfolded_x,
        dim=2,
        index=inds[:, None, :].expand(B, patch_size**2, N),
    )  # B x K_H * K_W x N
    return patches


# @torch.compile()
def sample_keypoints(
    keypoint_probs: torch.Tensor,
    num_samples=8192,
    use_nms=True,
    nms_size=1,
    sample_topk=True,
    increase_coverage=True,
    remove_borders=False,
    return_probs=False,
    coverage_pow=1 / 2,
    coverage_size=51,
    subpixel=False,
    scoremap=None,  # required for subpixel
    subpixel_temp=0.5,
    coverage_from_sparse=False,
):
    device = keypoint_probs.device
    B, H, W = keypoint_probs.shape
    if coverage_from_sparse:
        non_coverage_points, non_coverage_probs = sample_keypoints(
            keypoint_probs,
            num_samples=4 * num_samples,
            use_nms=use_nms,
            nms_size=nms_size,
            sample_topk=sample_topk,
            remove_borders=remove_borders,
            coverage_pow=coverage_pow,
            coverage_size=coverage_size,
            subpixel=subpixel,
            scoremap=scoremap,
            subpixel_temp=subpixel_temp,
            increase_coverage=False,
            coverage_from_sparse=False,
            return_probs=True,
        )
        density = kde(non_coverage_points)
        balance = 1 / (density + 1)
        balance[density < 10] = 1e-7
        p: torch.Tensor = non_coverage_probs * balance
        inds = p.argsort(dim=-1, descending=True)[:, :num_samples]
        # print(non_coverage_points.shape, inds.shape)
        kps = torch.gather(
            non_coverage_points,
            dim=1,
            index=inds[..., None].expand(B, num_samples, 2),
        )

        if return_probs:
            return kps, torch.gather(non_coverage_probs, dim=1, index=inds)
        return kps
    if increase_coverage and not coverage_from_sparse:
        weights = (
            -(torch.linspace(-2, 2, steps=coverage_size, device=device) ** 2)
        ).exp()[None, None]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d(
            (keypoint_probs[:, None] + 1e-6) * 10000,
            weights[..., None, :],
            padding=(0, coverage_size // 2),
        )
        local_density = F.conv2d(
            local_density_x, weights[..., None], padding=(coverage_size // 2, 0)
        )[:, 0]
        keypoint_probs = keypoint_probs * (local_density + 1e-8) ** (-coverage_pow)
    grid = get_normalized_grid(B, H, W, overload_device=device).reshape(B, H * W, 2)
    if use_nms:
        keypoint_probs = keypoint_probs * (
            keypoint_probs
            == F.max_pool2d(keypoint_probs, nms_size, stride=1, padding=nms_size // 2)
        )
    if remove_borders:
        frame = torch.zeros_like(keypoint_probs)
        # we hardcode 4px, could do it nicer, but whatever
        frame[..., 4:-4, 4:-4] = 1
        keypoint_probs = keypoint_probs * frame
    if sample_topk:
        inds = torch.topk(keypoint_probs.reshape(B, H * W), k=num_samples).indices
    else:
        inds = torch.multinomial(
            keypoint_probs.reshape(B, H * W), num_samples=num_samples, replacement=False
        )
    kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_samples, 2))
    if subpixel:
        if scoremap is None:
            raise ValueError("scoremap is required when subpixel=True")
        offsets = get_normalized_grid(
            B, nms_size, nms_size, overload_device=device
        ).reshape(B, nms_size**2, 2)  # B x K_H x K_W x 2
        offsets[..., 0] = offsets[..., 0] * nms_size / W
        offsets[..., 1] = offsets[..., 1] * nms_size / H
        keypoint_patch_scores = extract_patches_from_inds(scoremap, inds, nms_size)
        keypoint_patch_probs = (keypoint_patch_scores / subpixel_temp).softmax(
            dim=1
        )  # B x K_H * K_W x N
        keypoint_offsets = torch.einsum("bkn, bkd ->bnd", keypoint_patch_probs, offsets)
        kps = kps + keypoint_offsets
    if return_probs:
        return kps, torch.gather(keypoint_probs.reshape(B, H * W), dim=1, index=inds)
    return kps


def masked_log_softmax(logits, mask):
    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits[mask] = logits[mask]
    log_p = masked_logits.log_softmax(dim=-1)
    return log_p


def masked_softmax(logits, mask):
    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits[mask] = logits[mask]
    log_p = masked_logits.softmax(dim=-1)
    return log_p


def cross_entropy(log_p_hat: torch.Tensor, p: torch.Tensor):
    return -(log_p_hat * p).sum(dim=-1)


def kl_div(p: torch.Tensor, log_p_hat: torch.Tensor):
    return cross_entropy(log_p_hat, p) - cross_entropy((p + 1e-12).log(), p)


def draw_kpts(im, kpts, radius=2, width=1):
    im = np.array(im)
    # Convert keypoints to numpy array
    kpts_np = kpts.cpu().numpy()

    # Create a copy of the image to draw on
    ret = im.copy()

    # Define green color (BGR format in OpenCV)
    green_color = (0, 255, 0)

    # Draw green plus signs for each keypoint
    for x, y in kpts_np:
        # Convert to integer coordinates
        x, y = int(x), int(y)

        # Draw horizontal line of the plus sign
        cv2.line(ret, (x - radius, y), (x + radius, y), green_color, width)
        # Draw vertical line of the plus sign
        cv2.line(ret, (x, y - radius), (x, y + radius), green_color, width)

    return ret


def visualize_keypoints(img_path, vis_path, detector: Detector, num_keypoints: int):
    img_path, vis_path = Path(img_path), Path(vis_path).with_suffix(".png")
    img = Image.open(img_path).convert("RGB")
    detections = detector.detect_from_path(
        img_path, num_keypoints=num_keypoints, return_dense_probs=True
    )
    W, H = img.size
    kps = detections["keypoints"]
    kps = detector.to_pixel_coords(kps, H, W)
    (vis_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(draw_kpts(img, kps[0])).save(vis_path)
    if detections.get("dense_probs") is not None:
        tensor_to_pil(detections["dense_probs"].squeeze().cpu(), autoscale=True).save(
            vis_path.as_posix().replace(".png", "_dense_probs.png")
        )


def run_qualitative_examples(
    *, model: Detector, workspace_path: str | Path, test_num_keypoints: int, step: int
):
    workspace_path = Path(workspace_path)
    torch.cuda.empty_cache()
    for im_path in [
        "assets/0015_A.jpg",
        "assets/0015_B.jpg",
        "assets/0032_A.jpg",
        "assets/0032_B.jpg",
        "assets/apprentices.jpg",
        "assets/rectangles_and_circles.png",
    ]:
        visualize_keypoints(
            im_path,
            workspace_path / "vis" / str(step) / im_path,
            model,
            num_keypoints=test_num_keypoints,
        )
    torch.cuda.empty_cache()
