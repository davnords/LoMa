"""Export LoMa models to ONNX.

A LoMa "model" is a 3-stage pipeline:
  1. detector   (DaD)            : image            -> keypoints, probs      [shared by ALL variants]
  2. descriptor (DeDoDe-B / -G)  : image, keypoints -> descriptions          [B128->dedode_b, others->dedode_g]
  3. matcher    (transformer)    : kpts0,kpts1,desc0,desc1 -> matches        [per-variant weights]

We export each stage as a separate .onnx so they can be re-assembled (and so the
heavy detector/descriptor are shared instead of duplicated per variant).

Run on CPU in fp32 for a clean graph (autocast / bf16 disabled).
"""

import argparse
import os
import sys
from dataclasses import replace

# Allow forcing CPU via env (LOMA_EXPORT_CPU=1) before loma/device.py picks the
# global `device` at import time. Default: use the GPU if available.
if os.environ.get("LOMA_EXPORT_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loma.loma import (  # noqa: E402
    LoMa,
    LoMaB,
    LoMaB128,
    LoMaL,
    LoMaG,
    LoMaR,
    filter_matches,
)
from loma.descriptor import dedode as _dedode  # noqa: E402


# --------------------------------------------------------------------------- #
# Make the frozen DINOv2 (used by dedode_g) export-friendly.
# The original forward is wrapped in @torch.compiler.disable (opaque to
# torch.export) and runs under `with torch.inference_mode()` (produces inference
# tensors that torch.export rejects). Replace it with a plain forward.
# --------------------------------------------------------------------------- #
def _dino_forward_export(self, x):
    B, C, H, W = x.shape
    feats = self.dinov2_vitl14.forward_features(x.to(self.amp_dtype))
    features_16 = (
        feats["x_norm_patchtokens"].permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
    )
    return [features_16], [(H // 14, W // 14)]


_dedode.FrozenDINOv2.forward = _dino_forward_export

VARIANTS = {
    "B": LoMaB,
    "B128": LoMaB128,
    "R": LoMaR,
    "L": LoMaL,
    "G": LoMaG,
}

# descriptor arch -> which variants use it (descriptor is shared within an arch)
DESC_ARCH = {
    "B": "dedode_g",
    "B128": "dedode_b",
    "R": "dedode_g",
    "L": "dedode_g",
    "G": "dedode_g",
}

DESC_H = DESC_W = 784  # DeDoDeDescriptor.read_image default (what the pipeline uses)


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def prepare(module: nn.Module) -> nn.Module:
    """Put a (sub)module into a clean fp32, autocast-free, eval state for export."""
    module.eval()
    module.float()
    for m in module.modules():
        if hasattr(m, "amp"):
            m.amp = False
        if hasattr(m, "amp_dtype"):
            m.amp_dtype = torch.float32
    return module


def build_model(variant: str) -> LoMa:
    cfg = VARIANTS[variant]()
    # mp=False -> no autocast in matcher forward; compile=False -> no torch.compile wrappers.
    cfg = replace(cfg, mp=False, compile=False)
    log(f"building {variant} (input_dim={cfg.input_dim}, embed_dim={cfg.embed_dim}, "
        f"n_layers={cfg.n_layers}, heads={cfg.num_heads}, desc={cfg.descriptor})")
    model = LoMa(cfg)
    prepare(model)
    return model


# --------------------------------------------------------------------------- #
# Export wrappers
# --------------------------------------------------------------------------- #
class MatcherWrapper(nn.Module):
    def __init__(self, model: LoMa, filter_threshold: float):
        super().__init__()
        self.model = model
        self.th = float(filter_threshold)

    def forward(self, kpts0, kpts1, desc0, desc1):
        scores = self.model(kpts0, kpts1, desc0, desc1)["scores"]
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.th)
        return m0, m1, mscores0, mscores1


class DescriptorWrapper(nn.Module):
    def __init__(self, model: LoMa):
        super().__init__()
        self.desc = model._descriptor

    def forward(self, image, keypoints):
        grid = self.desc(image)
        described = F.grid_sample(
            grid.float(), keypoints[:, None], mode="bilinear", align_corners=False
        )[:, :, 0].transpose(1, 2)
        return described


class PlainNormalize(nn.Module):
    """(x - mean) / std without torchvision's data-dependent `if (std==0).any()`
    branch, which torch.export cannot trace."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class DetectorWrapper(nn.Module):
    def __init__(self, model: LoMa, num_keypoints: int):
        super().__init__()
        self.det = model._detector
        # Replace torchvision Normalize (data-dependent branch) with plain arithmetic.
        self.det.normalizer = PlainNormalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ).to(dev(model))
        self.num_keypoints = int(num_keypoints)

    def forward(self, image):
        out = self.det(image, self.num_keypoints, return_dense_probs=False)
        return out["keypoints"], out["keypoint_probs"]


# --------------------------------------------------------------------------- #
# Real sample inputs (for tracing + validation)
# --------------------------------------------------------------------------- #
def dev(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def norm(t: torch.Tensor, d: torch.device) -> torch.Tensor:
    """Detach+clone to escape inference-mode tensors (rejected by torch.export)."""
    return t.detach().clone().float().to(d)


def sample_image(model: LoMa, path: str) -> torch.Tensor:
    """Detector-preprocessed image tensor [1,3,H,W] in [0,1]."""
    return norm(model._detector.load_image(path), dev(model))


def sample_desc_image(model: LoMa, path: str) -> torch.Tensor:
    """Descriptor-preprocessed image tensor [1,3,784,784] in [0,1]."""
    return norm(model._descriptor.read_image(path, H=DESC_H, W=DESC_W), dev(model))


def real_matcher_inputs(model: LoMa, im_a: str, im_b: str, n: int):
    d = dev(model)
    ka, da, _, _ = model.detect_and_describe(im_a, num_keypoints=n)
    kb, db, _, _ = model.detect_and_describe(im_b, num_keypoints=n)
    return norm(ka, d), norm(kb, d), norm(da, d), norm(db, d)


# --------------------------------------------------------------------------- #
# ONNX export + validation helpers
# --------------------------------------------------------------------------- #
def consolidate(path):
    """Fold any sibling `<path>.data` external weights back into a single self-
    contained .onnx file (skipped if the model would exceed the 2GB protobuf limit)."""
    data = path + ".data"
    if not os.path.exists(data):
        return
    import onnx
    try:
        model = onnx.load(path)  # pulls in external data from the sibling file
        onnx.save_model(model, path, save_as_external_data=False)
        os.remove(data)
        log(f"  consolidated weights into single file ({os.path.getsize(path)/1e6:.1f} MB)")
    except Exception as e:
        log(f"  kept external weights ({os.path.getsize(data)/1e6:.1f} MB .data): "
            f"{type(e).__name__}: {str(e)[:120]}")


def do_export(wrapper, args_tuple, path, input_names, output_names, opset,
              dynamic_shapes=None, allow_legacy=True):
    """Export via the dynamo (torch.export) path. The legacy TorchScript exporter
    emits ORT-invalid graphs for these models (bad Concat axis, MaxPool dilations),
    so it is only a last resort. Tries dynamic shapes, then static, then legacy."""
    wrapper.eval()
    attempts = []
    if dynamic_shapes is not None:
        attempts.append(("dynamo+dynamic",
                         dict(dynamo=True, optimize=True, dynamic_shapes=dynamic_shapes)))
    attempts.append(("dynamo-static", dict(dynamo=True, optimize=True)))
    errs = []
    with torch.no_grad():
        for label, kw in attempts:
            try:
                torch.onnx.export(
                    wrapper, args_tuple, path,
                    input_names=input_names, output_names=output_names,
                    opset_version=opset, **kw,
                )
                consolidate(path)
                log(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB) [{label}]")
                return label
            except Exception as e:
                errs.append(f"{label}: {type(e).__name__}: {str(e)[:160]}")
                log(f"  {label} failed: {errs[-1]}")
        if not allow_legacy:
            raise RuntimeError("dynamo export failed: " + " | ".join(errs))
        # last resort: legacy
        torch.onnx.export(
            wrapper, args_tuple, path,
            input_names=input_names, output_names=output_names,
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    consolidate(path)
    log(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB) [legacy]")
    return "legacy"


def ort_session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def kpt_agreement(ref, got, tol=1e-3):
    """Order-independent keypoint-set agreement: fraction of `got` keypoints that
    have a `ref` keypoint within `tol` (normalized coords). topk/TopK return the
    same set in different order across backends, so element-wise compare is wrong."""
    fracs = []
    for b in range(ref.shape[0]):
        r, g = ref[b], got[b]  # [N,2]
        d = np.sqrt(((g[:, None, :] - r[None, :, :]) ** 2).sum(-1))  # [N,N]
        fracs.append(float((d.min(1) < tol).mean()))
    return float(np.mean(fracs))


def compare(name, ref, got, atol=1e-3, rtol=1e-3):
    ref = np.asarray(ref)
    got = np.asarray(got)
    if ref.shape != got.shape:
        log(f"  [validate] {name}: SHAPE MISMATCH ref{ref.shape} vs onnx{got.shape}")
        return False
    if ref.dtype.kind in "iu":  # integer (match indices) -> exact agreement ratio
        agree = float((ref == got).mean())
        log(f"  [validate] {name}: index agreement {agree*100:.2f}%")
        return agree > 0.99
    diff = np.abs(ref - got)
    ok = bool(np.allclose(ref, got, atol=atol, rtol=rtol))
    log(f"  [validate] {name}: max|diff|={diff.max():.2e} mean={diff.mean():.2e} "
        f"allclose={ok}")
    return ok


# --------------------------------------------------------------------------- #
# Per-component drivers
# --------------------------------------------------------------------------- #
def export_matcher(model, variant, outdir, opset, n, im_a, im_b):
    from torch.export import Dim
    log(f"== matcher {variant} ==")
    wrapper = prepare(MatcherWrapper(model, model.cfg.filter_threshold))
    ka, kb, da, db = real_matcher_inputs(model, im_a, im_b, n)
    path = os.path.join(outdir, f"loma_matcher_{variant}.onnx")
    # batch + per-side keypoint counts are dynamic; descriptor dim is static.
    b = Dim("batch")
    n0, n1 = Dim("n0", min=16), Dim("n1", min=16)
    dynamic_shapes = ({0: b, 1: n0}, {0: b, 1: n1}, {0: b, 1: n0}, {0: b, 1: n1})
    mode = do_export(
        wrapper, (ka, kb, da, db), path,
        ["kpts0", "kpts1", "desc0", "desc1"],
        ["m0", "m1", "mscores0", "mscores1"], opset, dynamic_shapes,
    )
    with torch.no_grad():
        ref = wrapper(ka, kb, da, db)
    sess = ort_session(path)
    got = sess.run(None, {
        "kpts0": to_np(ka), "kpts1": to_np(kb),
        "desc0": to_np(da), "desc1": to_np(db),
    })
    # Correctness is judged by match-index agreement (m0). mscores can differ at a
    # few entries due to argmax tie-flips (GPU torch vs CPU ORT) -- report, don't fail.
    m0_ref, m0_got = to_np(ref[0]), got[0]
    agree = float((m0_ref == m0_got).mean())
    ms_ref, ms_got = to_np(ref[2]), got[2]
    ms_flips = float((np.abs(ms_ref - ms_got) > 1e-2).mean())
    log(f"  [validate] m0 agreement {agree*100:.3f}% | "
        f"mscores0 >1e-2 diffs {ms_flips*100:.3f}% (mean|diff|={np.abs(ms_ref-ms_got).mean():.2e})")
    ok = agree > 0.99 and ms_flips < 0.01
    log(f"  matcher {variant} [{mode}]: {'OK' if ok else 'MISMATCH'}")
    return ok


def export_descriptor(model, arch, outdir, opset, im_b, n):
    from torch.export import Dim
    log(f"== descriptor {arch} ==")
    wrapper = prepare(DescriptorWrapper(model))
    img = sample_desc_image(model, im_b)
    kpts, _, _, _ = model.detect_and_describe(im_b, num_keypoints=n)
    kpts = norm(kpts, dev(model))
    path = os.path.join(outdir, f"loma_descriptor_{arch}.onnx")
    b = Dim("batch")
    nk = Dim("nkpts", min=16)
    dynamic_shapes = ({0: b}, {0: b, 1: nk})
    mode = do_export(wrapper, (img, kpts), path, ["image", "keypoints"],
                     ["descriptions"], opset, dynamic_shapes)
    with torch.no_grad():
        ref = wrapper(img, kpts)
    sess = ort_session(path)
    got = sess.run(None, {"image": to_np(img), "keypoints": to_np(kpts)})[0]
    ok = compare("descriptions", to_np(ref), got, atol=1e-2, rtol=1e-2)
    log(f"  descriptor {arch} [{mode}]: {'OK' if ok else 'MISMATCH'}")
    return ok


def export_detector(model, outdir, opset, n, im_b, dynamic):
    log("== detector (DaD, shared) ==")
    img = sample_image(model, im_b)
    path = os.path.join(outdir, "loma_detector.onnx")
    # Detector exports STATIC: dynamic batch hits `if B == 0` in get_normalized_grid,
    # and dynamic H/W hits data-dependent guards. (Re-export per input size if needed.)
    dynamic_shapes = None

    # subpixel refinement uses nn.Unfold (im2col); if torch.export won't trace it,
    # retry without it. allow_legacy=False so a dynamo failure raises (legacy MaxPool
    # is ORT-invalid) and we fall through to the next subpixel setting.
    last_err = None
    for subpixel in (True, False):
        model._detector.subpixel = subpixel
        wrapper = prepare(DetectorWrapper(model, n))
        try:
            mode = do_export(wrapper, (img,), path, ["image"],
                             ["keypoints", "keypoint_probs"], opset,
                             dynamic_shapes, allow_legacy=False)
            with torch.no_grad():
                ref_k, ref_p = wrapper(img)
            sess = ort_session(path)
            got = sess.run(None, {"image": to_np(img)})
            # keypoints are an unordered set -> order-independent agreement.
            agree = kpt_agreement(to_np(ref_k), got[0], tol=1e-3)
            ok_p = compare("keypoint_probs", to_np(ref_p), got[1], atol=1e-2, rtol=1e-2)
            log(f"  [validate] keypoint set agreement {agree*100:.2f}% (tol=1e-3)")
            ok = ok_p and agree > 0.98
            log(f"  detector (subpixel={subpixel}) [{mode}]: {'OK' if ok else 'MISMATCH'}")
            if ok:
                if not subpixel:
                    log("  NOTE: subpixel refinement disabled in ONNX.")
                return True
            if subpixel:
                log("  retrying detector without subpixel...")
        except Exception as e:
            last_err = e
            log(f"  detector subpixel={subpixel} failed: {type(e).__name__}: {str(e)[:160]}")
            continue
    if last_err:
        raise last_err
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["B", "B128", "R"],
                    choices=list(VARIANTS))
    ap.add_argument("--components", nargs="+", default=["matcher", "descriptor", "detector"],
                    choices=["matcher", "descriptor", "detector"])
    ap.add_argument("--outdir", default="onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--num-keypoints", type=int, default=2048)
    ap.add_argument("--im-a", default="assets/0015_A.jpg")
    ap.add_argument("--im-b", default="assets/0015_B.jpg")
    ap.add_argument("--detector-dynamic", action="store_true",
                    help="export detector with dynamic H/W (else fixed to sample size)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n = args.num_keypoints
    results = {}
    done_desc = set()
    detector_done = False

    for v in args.variants:
        model = build_model(v)
        if "matcher" in args.components:
            try:
                results[f"matcher_{v}"] = export_matcher(
                    model, v, args.outdir, args.opset, n, args.im_a, args.im_b)
            except Exception as e:
                results[f"matcher_{v}"] = False
                log(f"  matcher {v} FAILED: {type(e).__name__}: {e}")
        if "descriptor" in args.components:
            arch = DESC_ARCH[v]
            if arch not in done_desc:
                try:
                    results[f"descriptor_{arch}"] = export_descriptor(
                        model, arch, args.outdir, args.opset, args.im_b, n)
                except Exception as e:
                    results[f"descriptor_{arch}"] = False
                    log(f"  descriptor {arch} FAILED: {type(e).__name__}: {e}")
                done_desc.add(arch)
        if "detector" in args.components and not detector_done:
            try:
                results["detector"] = export_detector(
                    model, args.outdir, args.opset, n, args.im_b, args.detector_dynamic)
            except Exception as e:
                results["detector"] = False
                log(f"  detector FAILED: {type(e).__name__}: {e}")
            detector_done = True
        del model

    log("==================== SUMMARY ====================")
    for k, ok in results.items():
        log(f"  {k:24s} {'OK' if ok else 'FAIL'}")
    bad = [k for k, ok in results.items() if not ok]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
