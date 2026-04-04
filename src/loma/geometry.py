from __future__ import annotations
from typing import Literal
import cv2
import torch
import torch.nn.functional as F
from einops import einsum
import numpy as np

from loma.device import device
from loma.types import Warp, GTSource, Batch


def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)


def from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return x[..., :-1] / x[..., -1:]


def get_normalized_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    if B == 0:
        return torch.zeros(0, H, W, 2, device=overload_device or device)
    x1_n = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=overload_device or device)
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def get_pixel_grid(
    B: int,
    *,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[torch.arange(n, device=overload_device or device) + 0.5 for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def to_normalized(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    assert x.shape[-1] == 2, "x must have shape (..., 2)"
    return torch.stack((2 * x[..., 0] / W, 2 * x[..., 1] / H), dim=-1) - 1


def to_pixel(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    assert x.shape[-1] == 2, "x must have shape (..., 2)"
    return torch.stack(((x[..., 0] + 1) / 2 * W, (x[..., 1] + 1) / 2 * H), dim=-1)


def _pixel_warp_and_depth_from_depth(
    *,
    pixel_coords_A: torch.Tensor,
    depth_A: torch.Tensor,
    K_A: torch.Tensor,
    K_B: torch.Tensor,
    T_AB: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    homog_pixel_coords_A = to_homogeneous(pixel_coords_A)
    x = einsum(
        homog_pixel_coords_A,
        K_A.inverse(),
        "batch H W pixel, batch calib pixel -> batch H W calib",
    )
    x = x * depth_A
    x = to_homogeneous(x)
    x = einsum(x, T_AB, "batch H W calib, batch world calib -> batch H W world")
    x_AB = from_homogeneous(x)
    x = einsum(x_AB, K_B, "batch H W world, batch pixel world -> batch H W pixel")
    pixel_coords_AB = from_homogeneous(x)
    depth = x_AB[..., -1:]
    return pixel_coords_AB, depth


def warp_and_depth_consistency_from_depths(
    *,
    depth_A: torch.Tensor,
    depth_B: torch.Tensor,
    K_A: torch.Tensor,
    K_B: torch.Tensor,
    T_AB: torch.Tensor,
    rel_depth_error_threshold: float,
    interp_rtol: float | None = None,
    interp_atol: float | None = None,
    pos_depth_check_bugfix: bool = True,
    local_neighbourhood_size: int = 1,
) -> Warp:
    B, H_B, W_B, one = depth_B.shape
    assert one == 1, "depth_A and depth_B must have shape (B, H, W, 1)"
    B, H_A, W_A, one = depth_A.shape
    assert one == 1, "depth_A and depth_B must have shape (B, H, W, 1)"
    pixel_coords_A = get_pixel_grid(B, H=H_A, W=W_A, overload_device=device)
    pixel_coords_AB, z_AB = _pixel_warp_and_depth_from_depth(
        pixel_coords_A=pixel_coords_A,
        depth_A=depth_A,
        K_A=K_A,
        K_B=K_B,
        T_AB=T_AB,
    )

    normalized_coords_AB = to_normalized(pixel_coords_AB, H=H_B, W=W_B)

    # Compute covisibility by checking local neighborhood
    K = local_neighbourhood_size
    K2 = K * K
    pad_size = K // 2

    # Pad and unfold to get all local neighbors of depth_B
    # depth_B: (B, H_B, W_B, 1) -> (B, 1, H_B, W_B)
    depth_B_bchw = depth_B.permute(0, 3, 1, 2)
    pad_depth_B = F.pad(
        depth_B_bchw, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
    )
    # unfold: (B, 1, H_B+2*pad, W_B+2*pad) -> (B, 1, H_B, W_B, K, K)
    unfolded = pad_depth_B.unfold(2, K, 1).unfold(3, K, 1)
    # Reshape to (B*K2, H_B, W_B, 1) by stacking neighbors in batch dimension
    unfolded = (
        unfolded.reshape(B, 1, H_B, W_B, K2)
        .permute(0, 4, 2, 3, 1)
        .reshape(B * K2, H_B, W_B, 1)
    )

    # Repeat sampling coordinates and depth values K2 times in batch dimension
    normalized_coords_AB_repeated = normalized_coords_AB.repeat_interleave(K2, dim=0)
    z_AB_repeated = z_AB.repeat_interleave(K2, dim=0)
    depth_A_repeated = depth_A.repeat_interleave(K2, dim=0)

    # Sample from unfolded depth_B at warped coordinates
    z_B = bhwc_grid_sample_with_nearest_exact_fallback(
        x=unfolded,
        grid=normalized_coords_AB_repeated,
        rtol=interp_rtol,
        atol=interp_atol,
        mode="nearest",
        align_corners=False,
    )

    # Compute consistency for all neighbors
    if pos_depth_check_bugfix:
        pos_and_finite_depth = (
            (depth_A_repeated > 0.0)
            .logical_and(depth_A_repeated.isfinite())
            .logical_and(z_AB_repeated > 0.0)
            .logical_and(z_AB_repeated.isfinite())
            .logical_and(z_B > 0.0)
            .logical_and(z_B.isfinite())
        )
    else:
        pos_and_finite_depth = (
            (z_AB_repeated > 0.0)
            .logical_and(z_AB_repeated.isfinite())
            .logical_and(~z_B.isnan())
        )

    rel_depth_error = ((z_B - z_AB_repeated) / z_B).abs()

    consistent_depth = (rel_depth_error < rel_depth_error_threshold).logical_and(
        pos_and_finite_depth
    )

    # Reshape and aggregate over all neighbors: (B*K2, H_A, W_A, 1) -> (B, K2, H_A, W_A, 1) -> (B, H_A, W_A, 1)
    covis_bool = consistent_depth.reshape(B, K2, H_A, W_A, 1).any(dim=1)
    rel_depth_error = rel_depth_error.reshape(B, K2, H_A, W_A, 1).amin(dim=1)
    pos_and_finite_depth = pos_and_finite_depth.reshape(B, K2, H_A, W_A, 1).any(dim=1)

    covis = covis_bool.float()
    pos_depth_and_within_frame = (pos_and_finite_depth).logical_and(
        normalized_coords_AB.abs().amax(dim=-1, keepdim=True) < 1.0
    )
    # Always compute valid (equivalent to mode "frame")
    valid = pos_depth_and_within_frame.float()
    # if rel_depth_error.isnan().any():
    #     print("rel_depth_error is nan")
    return Warp(
        warp=normalized_coords_AB, covis=covis, valid=valid, error=rel_depth_error
    )


def warp_and_cycle_consistency_from_depths(
    *,
    depth_A: torch.Tensor,
    depth_B: torch.Tensor,
    K_A: torch.Tensor,
    K_B: torch.Tensor,
    T_AB: torch.Tensor,
    cycle_error_threshold: float,
    interp_rtol: float | None = None,
    interp_atol: float | None = None,
    multi_hyp: bool = False,
) -> Warp:
    B, H_A, W_A, _ = depth_A.shape
    B, H_B, W_B, _ = depth_B.shape

    pixel_coords_A = get_pixel_grid(B, H=H_A, W=W_A, overload_device=device)
    pixel_coords_AB, depth_AB = _pixel_warp_and_depth_from_depth(
        pixel_coords_A=pixel_coords_A,
        depth_A=depth_A,
        K_A=K_A,
        K_B=K_B,
        T_AB=T_AB,
    )
    normalized_coords_AB = to_normalized(pixel_coords_AB, H=H_B, W=W_B)
    depth_AB_B = bhwc_grid_sample_with_nearest_exact_fallback(
        x=depth_B,
        grid=normalized_coords_AB,
        rtol=interp_rtol,
        atol=interp_atol,
        mode="bilinear",
        align_corners=False,
    )
    pos_and_finite_depth = (depth_AB_B > 0.0).logical_and(depth_AB_B.isfinite())
    pos_depth_and_within_frame = (pos_and_finite_depth).logical_and(
        normalized_coords_AB.abs().amax(dim=-1, keepdim=True) < 1.0
    )
    valid = pos_depth_and_within_frame.float()
    pixel_coords_AB_BA, _ = _pixel_warp_and_depth_from_depth(
        pixel_coords_A=pixel_coords_AB,
        depth_A=depth_AB_B,
        K_A=K_B,
        K_B=K_A,
        T_AB=T_AB.inverse(),
    )
    cycle_pixel_error = (pixel_coords_AB_BA - pixel_coords_A).norm(dim=-1, keepdim=True)
    cycle_consistent = (cycle_pixel_error < cycle_error_threshold).logical_and(
        pos_and_finite_depth
    )
    return Warp(
        warp=normalized_coords_AB,
        covis=cycle_consistent.float(),
        valid=valid,
        error=cycle_pixel_error,
    )


def bhwc_interpolate(
    x: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool | None = None,
    antialias: bool | None = None,
) -> torch.Tensor:
    x_chw = x.permute(0, 3, 1, 2)
    if align_corners is not None and antialias is not None:
        y = F.interpolate(
            x_chw,
            size=size,
            mode=mode,
            align_corners=align_corners,
            antialias=antialias,
        )
    elif align_corners is not None:
        y = F.interpolate(x_chw, size=size, mode=mode, align_corners=align_corners)
    elif antialias is not None:
        y = F.interpolate(x_chw, size=size, mode=mode, antialias=antialias)
    else:
        y = F.interpolate(x_chw, size=size, mode=mode)
    return y.permute(0, 2, 3, 1)


def bhwc_interpolate_with_nearest_exact_fallback(
    x: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> torch.Tensor:
    x_interp = bhwc_interpolate(x, size, mode, align_corners)
    if rtol is None and atol is None:
        return x_interp
    x_nearest = bhwc_interpolate(x, size, "nearest-exact")
    on_boundary = torch.isclose(
        x_interp,
        x_nearest,
        rtol=rtol if rtol is not None else 0.0,
        atol=atol if atol is not None else 0.0,
    ).logical_not()
    x_interp[on_boundary] = x_nearest[on_boundary]
    return x_interp


def bhwc_grid_sample(
    x: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool | None = False,
) -> torch.Tensor:
    return F.grid_sample(
        x.permute(0, 3, 1, 2), grid, mode=mode, align_corners=align_corners
    ).permute(0, 2, 3, 1)


def bhwc_grid_sample_with_nearest_exact_fallback(
    *,
    x: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> torch.Tensor:
    x_interp = F.grid_sample(
        x.permute(0, 3, 1, 2), grid, mode=mode, align_corners=align_corners
    ).permute(0, 2, 3, 1)
    if rtol is None and atol is None:
        return x_interp
    x_nearest = F.grid_sample(
        x.permute(0, 3, 1, 2), grid, mode="nearest", align_corners=align_corners
    ).permute(0, 2, 3, 1)
    x = x_interp
    on_boundary = torch.isclose(
        x_interp,
        x_nearest,
        rtol=rtol if rtol is not None else 0.0,
        atol=atol if atol is not None else 0.0,
    ).logical_not()
    # at motion boundaries we fall back to nearest-exact
    # note, this is pointwise
    x[on_boundary] = x_nearest[on_boundary]
    return x


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(R_gt, t_gt, R, t):
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapezoid(r, x=e) / t)  # type: ignore
    return aucs


def cov_mat_from_cov_params(c: torch.Tensor) -> torch.Tensor:
    return prec_mat_from_prec_params(c)


def prec_mat_from_prec_params(p: torch.Tensor) -> torch.Tensor:
    P = p.new_zeros(p.shape[0], p.shape[1], p.shape[2], 2, 2)
    P[..., 0, 0] = p[..., 0]
    P[..., 1, 0] = p[..., 1]
    P[..., 0, 1] = p[..., 1]
    P[..., 1, 1] = p[..., 2]
    return P


def to_double_angle_rep(v: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(v[..., 1], v[..., 0])
    double_angle_rep = torch.stack((torch.cos(2 * angle), torch.sin(2 * angle)), dim=-1)
    return double_angle_rep


def prec_mat_to_flow(
    P: torch.Tensor, vis_max: float, mode: Literal["smallest", "largest"] = "largest"
) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(P)
    if mode == "smallest":
        vis_vec = vecs[..., 0]
    elif mode == "largest":
        vis_vec = vecs[..., -1]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    # select upper half-plane
    vis_vec = torch.where(vis_vec[..., 1:2] >= 0, vis_vec, -vis_vec)
    double_angle_rep = to_double_angle_rep(vis_vec)
    scale = (vals[..., 0] * vals[..., 1]).pow(0.25).clamp(0, vis_max)[..., None]
    flow = scale * double_angle_rep
    return flow


def prec_params_to_flow(
    p: torch.Tensor, vis_max: float, mode: Literal["smallest", "largest"] = "largest"
) -> torch.Tensor:
    P = prec_mat_from_prec_params(p)
    return prec_mat_to_flow(P, vis_max, mode)


def flow_forward_backward_error(
    flow_AB: torch.Tensor,
    flow_BA: torch.Tensor,
    *,
    interp_rtol: float | None = None,
    interp_atol: float | None = None,
) -> torch.Tensor:
    B, H, W, _ = flow_AB.shape
    grid = get_normalized_grid(B, H, W, overload_device=flow_AB.device)
    fwd_bwd = bhwc_grid_sample_with_nearest_exact_fallback(
        x=flow_BA,
        grid=grid + flow_AB,
        rtol=interp_rtol,
        atol=interp_atol,
        mode="nearest",
        # mode="bilinear",
        # align_corners=False,
    )
    fwd_bwd_error = (flow_AB + fwd_bwd).norm(dim=-1, keepdim=True)
    return fwd_bwd_error


def warp_and_overlap_from_flows(
    *,
    flow_AB: torch.Tensor,
    flow_BA: torch.Tensor,
    interp_rtol: float | None = None,
    interp_atol: float | None = None,
    error_threshold: float,
    # check_local_neighbourhood: bool = False,
    local_neighbourhood_size: int = 1,
) -> Warp:
    # flow_AB and flow_BA are flows (offsets), convert to warps for processing
    B, H, W, _ = flow_AB.shape
    grid = get_normalized_grid(B, H, W, overload_device=flow_AB.device)
    grid_A = grid
    # grid_B = grid

    warp_AB = grid_A + flow_AB
    # Check if warp is in normalized coordinates [-1, 1]

    in_frame = warp_AB.abs().le(1.0).all(dim=-1, keepdim=True)

    # Compute covisibility by checking local neighborhood
    # if check_local_neighbourhood:
    K = local_neighbourhood_size
    K2 = K * K
    pad_size = K // 2

    # Pad and unfold to get all local neighbors
    # flow_BA: (B, H, W, 2) -> (B, 2, H, W)
    flow_BA_bchw = flow_BA.permute(0, 3, 1, 2)
    pad_flow_BA = F.pad(
        flow_BA_bchw, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
    )
    # unfold: (B, 2, H+2*pad, W+2*pad) -> (B, 2, H, W, K, K)
    unfolded = pad_flow_BA.unfold(2, K, 1).unfold(3, K, 1)
    # Reshape to (B*K2, H, W, 2) by stacking neighbors in batch dimension
    unfolded = (
        unfolded.reshape(B, 2, H, W, K2).permute(0, 4, 2, 3, 1).reshape(B * K2, H, W, 2)
    )

    # Repeat flow_AB K2 times in batch dimension (interleaved to match unfolded order)
    flow_AB_repeated = flow_AB.repeat_interleave(K2, dim=0)

    # Compute forward-backward error for all neighbors at once
    fwd_bwd_error = flow_forward_backward_error(
        flow_AB_repeated, unfolded, interp_rtol=interp_rtol, interp_atol=interp_atol
    )
    covis = fwd_bwd_error < error_threshold

    # Reshape and take logical_or over all neighbors: (B*K2, H, W, 1) -> (B, K2, H, W, 1) -> (B, H, W, 1)
    overlap_covis_bool = covis.reshape(B, K2, H, W, 1).any(dim=1).logical_and(in_frame)
    fwd_bwd_error = fwd_bwd_error.reshape(B, K2, H, W, 1).amin(dim=1) * in_frame.float()
    # else:
    #     fwd_bwd_error = flow_forward_backward_error(
    #         flow_AB, flow_BA, interp_rtol=interp_rtol, interp_atol=interp_atol
    #     )
    #     overlap_covis_bool = (fwd_bwd_error < error_threshold).logical_and(in_frame)

    return Warp(
        warp=warp_AB,
        covis=overlap_covis_bool.float(),
        valid=in_frame.float(),
        error=fwd_bwd_error,
    )


def compute_gt_warp_from_batch(
    batch: Batch,
    *,
    depth_error_threshold: float,
    flow_error_threshold: float,
    local_neighbourhood_size: int,
) -> Warp:
    assert isinstance(batch.source, list), "source must be a list"
    sources: list[GTSource] = batch.source

    is_flow_source = torch.tensor(
        [source == "flow" for source in sources], device=device
    )
    B, three, H_A, W_A = batch.img_A.shape
    B, three, H_B, W_B = batch.img_B.shape
    depth_A = batch.depth_A[~is_flow_source]
    depth_B = batch.depth_B[~is_flow_source]
    K_A = batch.K_A[~is_flow_source]
    K_B = batch.K_B[~is_flow_source]
    T_AB = batch.T_AB[~is_flow_source]
    flow_flow_source_AB = batch.flow_AB[is_flow_source]
    flow_flow_source_BA = batch.flow_BA[is_flow_source]

    result_depth = warp_and_depth_consistency_from_depths(
        depth_A=depth_A,
        depth_B=depth_B,
        K_A=K_A,
        K_B=K_B,
        T_AB=T_AB,
        rel_depth_error_threshold=depth_error_threshold,
        local_neighbourhood_size=local_neighbourhood_size,
    )
    overlap_depth_src = result_depth.covis
    warp_depth_src_AB = result_depth.warp
    valid_depth_src = result_depth.valid
    error_depth_src = result_depth.error

    result_flow = warp_and_overlap_from_flows(
        flow_AB=flow_flow_source_AB,
        flow_BA=flow_flow_source_BA,
        error_threshold=flow_error_threshold,
        local_neighbourhood_size=local_neighbourhood_size,
    )
    overlap_flow_src = result_flow.covis
    warp_flow_src_AB = result_flow.warp
    valid_flow_src = result_flow.valid
    error_flow_src = result_flow.error
    assert overlap_flow_src is not None and overlap_depth_src is not None
    assert warp_flow_src_AB is not None and warp_depth_src_AB is not None
    assert valid_flow_src is not None and valid_depth_src is not None
    assert error_flow_src is not None and error_depth_src is not None

    covis = torch.zeros((B, H_A, W_A, 1), dtype=torch.float32, device=device)
    covis[is_flow_source] = overlap_flow_src.float()
    covis[~is_flow_source] = overlap_depth_src.float()

    warp = torch.zeros((B, H_A, W_A, 2), device=device, dtype=torch.float32)
    warp[is_flow_source] = warp_flow_src_AB
    warp[~is_flow_source] = warp_depth_src_AB
    warp[~warp.isfinite()] = 0

    valid = torch.zeros((B, H_A, W_A, 1), device=device, dtype=torch.float32)
    valid[is_flow_source] = valid_flow_src
    valid[~is_flow_source] = valid_depth_src

    error = torch.zeros((B, H_A, W_A, 1), device=device, dtype=torch.float32)
    error[is_flow_source] = error_flow_src
    error[~is_flow_source] = error_depth_src

    return Warp(warp=warp, covis=covis, valid=valid, error=error)


# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose_cv2_ransac(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    norm_thresh: float,
    conf: float = 0.99999,
):
    import cv2

    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) // 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)  # type: ignore
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    assert ret is not None
    return ret


def compute_pose_inliers_cv2_ransac(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    norm_thresh: float,
    conf: float = 0.99999,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """Inlier ratio and best pose from the same RANSAC/E hypothesis as `estimate_pose_cv2_ransac`."""
    import cv2

    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    if E is None:
        return None
    best_num_inliers = 0
    ret: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for _E in np.split(E, len(E) // 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)  # type: ignore
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t, mask.ravel() > 0)
    assert ret is not None
    return (best_num_inliers / len(kpts0), ret)


def estimate_pose_essential(
    kps_A: np.ndarray,
    kps_B: np.ndarray,
    w_A: int,
    h_A: int,
    K_A: np.ndarray,
    w_B: int,
    h_B: int,
    K_B: np.ndarray,
    th: float,
) -> tuple[np.ndarray, np.ndarray]:
    import poselib

    camera1 = {
        "model": "PINHOLE",
        "width": w_A,
        "height": h_A,
        "params": K_A[[0, 1, 0, 1], [0, 1, 2, 2]],
    }
    camera2 = {
        "model": "PINHOLE",
        "width": w_B,
        "height": h_B,
        "params": K_B[[0, 1, 0, 1], [0, 1, 2, 2]],
    }

    pose, res = poselib.estimate_relative_pose(
        kps_A,
        kps_B,
        camera1,
        camera2,
        ransac_opt={
            "max_epipolar_error": th,
        },
    )
    return pose.R, pose.t


def poselib_fundamental(x1, x2, opt):
    import poselib

    F, info = poselib.estimate_fundamental(x1, x2, opt, {})
    inl = info["inliers"]
    return F, inl


def calibrate(x: torch.Tensor, K: torch.Tensor):
    # x: ..., 2
    # K: ..., 3, 3
    return to_homogeneous(x) @ K.inverse().mT


def estimate_pose_fundamental(
    kps_A: np.ndarray,
    kps_B: np.ndarray,
    w_A: int,
    h_A: int,
    K_A: np.ndarray,
    w_B: int,
    h_B: int,
    K_B: np.ndarray,
    th: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(kps_A) < 8:
        return np.eye(3), np.zeros(3)
    F, inl = poselib_fundamental(
        kps_A,
        kps_B,
        opt={
            "max_epipolar_error": th,
        },
    )
    E: np.ndarray = K_B.T @ F @ K_A
    kps_calib_A = from_homogeneous(
        calibrate(torch.from_numpy(kps_A).float(), torch.from_numpy(K_A).float())
    ).numpy()
    kps_calib_B = from_homogeneous(
        calibrate(torch.from_numpy(kps_B).float(), torch.from_numpy(K_B).float())
    ).numpy()
    E = E.astype(np.float64)
    _, R, t, good = cv2.recoverPose(E, kps_calib_A, kps_calib_B)
    t = t[:, 0]
    return R, t


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.mT)
    trans = -rots @ t1 + t2
    return rots, trans


def kde(x: torch.Tensor, std: float = 0.1, half: bool = True) -> torch.Tensor:
    # use a gaussian kernel to estimate density
    if half:
        x = x.half()
    scores = (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density


def normalize(x: torch.Tensor, dim: int):
    return x / x.norm(dim=dim, keepdim=True)


def cosine_similarity(f_A: torch.Tensor, f_B: torch.Tensor) -> torch.Tensor:
    # TODO: remove this?
    f_A = normalize(f_A, dim=-1)
    f_B = normalize(f_B, dim=-1)
    res = einsum(f_A, f_B, "B H_A W_A D, B H_B W_B D -> B H_A W_A H_B W_B")
    return res


@torch.no_grad()
def compute_sparse_mnn_matches(
    kpts_A,
    kpts_B,
    batch,
    *,
    depth_error_threshold: float,
    flow_error_threshold: float,
    local_neighbourhood_size: int,
    error_threshold: float,
    gt_AB: Warp | None = None,
    gt_BA: Warp | None = None,
):
    """
    Compute mutual nearest neighbor matches.

    For samples with source == "sparse", uses correspondences directly (keypoints should be
    [correspondences, detector_keypoints] concatenated, and MNN points to correspondence indices).

    For samples with source == "depth" or "flow", computes MNN using depth/flow warping.

    Args:
        gt_AB: Optional precomputed warp from A to B. If provided, skips recomputation.
        gt_BA: Optional precomputed warp from B to A. If provided, skips recomputation.
    """
    B = kpts_A.shape[0]
    device = kpts_A.device

    source = batch.source if isinstance(batch.source, list) else [batch.source] * B
    num_corresp = (
        batch.num_corresp
        if isinstance(batch.num_corresp, list)
        else [batch.num_corresp] * B
    )

    is_sparse = torch.tensor([s == "sparse" for s in source], device=device)
    all_mnn = []

    # Handle sparse samples: MNN points to correspondence indices (idx_A == idx_B)
    if is_sparse.any():
        sparse_counts = torch.tensor(
            [num_corresp[b] if is_sparse[b] else 0 for b in range(B)],
            device=device,
            dtype=torch.long,
        )
        total_sparse = sparse_counts.sum()
        if total_sparse > 0:
            batch_idx = torch.repeat_interleave(
                torch.arange(B, device=device), sparse_counts
            )
            corresp_idx = torch.cat(
                [torch.arange(n, device=device) for n in sparse_counts.tolist()]
            )
            sparse_mnn = torch.stack([batch_idx, corresp_idx, corresp_idx], dim=1)
            all_mnn.append(sparse_mnn)

    # Handle warp-based samples (depth/flow)
    if (~is_sparse).any():
        # Use precomputed warps if provided, otherwise compute them
        if gt_AB is None:
            gt_AB = compute_gt_warp_from_batch(
                batch,
                depth_error_threshold=depth_error_threshold,
                flow_error_threshold=flow_error_threshold,
                local_neighbourhood_size=local_neighbourhood_size,
            )
        if gt_BA is None:
            gt_BA = compute_gt_warp_from_batch(
                batch.swap_AB(),
                depth_error_threshold=depth_error_threshold,
                flow_error_threshold=flow_error_threshold,
                local_neighbourhood_size=local_neighbourhood_size,
            )
        kpts_A_to_B = bhwc_grid_sample(gt_AB.warp, kpts_A[:, None])[:, 0]  # (B, N, 2)
        kpts_B_to_A = bhwc_grid_sample(gt_BA.warp, kpts_B[:, None])[:, 0]  # (B, N, 2)
        D_B = torch.cdist(kpts_A_to_B, kpts_B).nan_to_num(nan=float("inf"))
        D_A = torch.cdist(kpts_A, kpts_B_to_A).nan_to_num(nan=float("inf"))
        warp_mnn = torch.nonzero(
            (D_B == D_B.min(dim=-1, keepdim=True).values)
            * (D_A == D_A.min(dim=-2, keepdim=True).values)
            * (D_B < error_threshold)
            * (D_A < error_threshold)
        )
        # Filter to only keep non-sparse samples
        if warp_mnn.numel() > 0:
            warp_mnn = warp_mnn[~is_sparse[warp_mnn[:, 0]]]
            if warp_mnn.numel() > 0:
                all_mnn.append(warp_mnn)

    if all_mnn:
        return torch.cat(all_mnn, dim=0)
    return torch.zeros((0, 3), device=device, dtype=torch.long)
