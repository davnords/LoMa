"""Export resolution/keypoint-optimized ONNX presets for Jetson Orin Nano.

Detector resolution + keypoint count are baked into the ONNX graph, so we emit one
detector and one descriptor per (preset). Matchers are dynamic and preset-independent
(exported by export_onnx.py, shared across presets).

Presets (square inputs for simple deployment; re-export with --det/--desc to match a
specific camera aspect ratio):
  fast     : detector 512, descriptor 512/518, 1024 keypoints
  balanced : detector 640, descriptor 640/630, 1536 keypoints
  quality  : detector 1024, descriptor 784,    2048 keypoints

dedode_g (DINOv2) requires descriptor side divisible by 14.
"""

import argparse
import os

import numpy as np
import torch

import export_onnx as E

# desc resolution per arch must satisfy: dedode_g -> divisible by 14.
# `det` is a square side (int) or a (H, W) tuple for non-square / landscape inputs.
PRESETS = {
    # ultra-fast presets for Jetson Orin Nano real-time (dedode_g side must be /14)
    "pico":     dict(det=256,         desc={"dedode_b": 256, "dedode_g": 252}, kpts=512),
    "nano":     dict(det=256,         desc={"dedode_b": 256, "dedode_g": 252}, kpts=1024),
    "turbo":    dict(det=384,         desc={"dedode_b": 384, "dedode_g": 378}, kpts=1024),
    "fast":     dict(det=512,         desc={"dedode_b": 512, "dedode_g": 518}, kpts=1024),
    "balanced": dict(det=640,         desc={"dedode_b": 640, "dedode_g": 630}, kpts=1536),
    "quality":  dict(det=1024,        desc={"dedode_b": 784, "dedode_g": 784}, kpts=2048),
    # landscape camera-frame preset: detector at H=512, W=1024
    "wide":     dict(det=(512, 1024), desc={"dedode_b": 512, "dedode_g": 518}, kpts=2048),
}

# one model per descriptor arch is enough to source both detector + descriptor weights
ARCH_VARIANT = {"dedode_b": "B128", "dedode_g": "B"}


def as_hw(v):
    """int -> (v, v) square; (H, W) tuple -> itself."""
    return (v, v) if isinstance(v, int) else tuple(v)


def preprocess_square(path, side, device):
    H, W = as_hw(side)
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((W, H))  # PIL resize takes (W, H)
    arr = np.array(im) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).float()[None].to(device)


def export_detector_preset(model, preset, outdir, opset, im):
    cfg = PRESETS[preset]
    H, W = as_hw(cfg["det"]); n = cfg["kpts"]
    path = os.path.join(outdir, f"loma_detector_{preset}.onnx")
    E.log(f"== detector [{preset}] {H}x{W} (HxW), {n} kpts ==")
    img = preprocess_square(im, (H, W), E.dev(model))
    last = None
    for subpixel in (True, False):
        model._detector.subpixel = subpixel
        w = E.prepare(E.DetectorWrapper(model, n))
        try:
            mode = E.do_export(w, (img,), path, ["image"],
                               ["keypoints", "keypoint_probs"], opset,
                               dynamic_shapes=None, allow_legacy=False)
            with torch.no_grad():
                rk, rp = w(img)
            s = E.ort_session(path)
            got = s.run(None, {"image": E.to_np(img)})
            agree = E.kpt_agreement(E.to_np(rk), got[0], tol=1e-3)
            okp = E.compare("keypoint_probs", E.to_np(rp), got[1], atol=1e-2, rtol=1e-2)
            E.log(f"  [validate] keypoint set agreement {agree*100:.2f}%")
            ok = okp and agree > 0.98
            E.log(f"  detector [{preset}] (subpixel={subpixel}) [{mode}]: "
                  f"{'OK' if ok else 'MISMATCH'}")
            if ok:
                return ("detector", preset, f"{H}x{W}", n, ok)
        except Exception as e:
            last = e
            E.log(f"  detector [{preset}] subpixel={subpixel} failed: "
                  f"{type(e).__name__}: {str(e)[:140]}")
    if last:
        raise last
    return ("detector", preset, f"{H}x{W}", n, False)


def export_descriptor_preset(model, arch, preset, outdir, opset, im):
    from torch.export import Dim
    side = PRESETS[preset]["desc"][arch]
    n = PRESETS[preset]["kpts"]
    path = os.path.join(outdir, f"loma_descriptor_{arch}_{preset}.onnx")
    E.log(f"== descriptor {arch} [{preset}] {side}x{side} ==")
    img = preprocess_square(im, side, E.dev(model))
    # deterministic pseudo-random normalized keypoints (descriptor is just grid_sample)
    g = torch.Generator().manual_seed(0)
    kpts = (2 * torch.rand(1, n, 2, generator=g) - 1).float().to(E.dev(model))
    w = E.prepare(E.DescriptorWrapper(model))
    b, nk = Dim("batch"), Dim("nkpts", min=16)
    mode = E.do_export(w, (img, kpts), path, ["image", "keypoints"],
                       ["descriptions"], opset, dynamic_shapes=({0: b}, {0: b, 1: nk}))
    with torch.no_grad():
        ref = w(img, kpts)
    s = E.ort_session(path)
    got = s.run(None, {"image": E.to_np(img), "keypoints": E.to_np(kpts)})[0]
    # Judge by cosine similarity (what matching actually uses) -- robust to the
    # occasional large abs outlier from grid_sample on random border keypoints.
    r = E.to_np(ref).astype(np.float64)
    g = got.astype(np.float64)
    rn = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-9)
    gn = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)
    cos = (rn * gn).sum(-1)
    maxd, meand = float(np.abs(r - g).max()), float(np.abs(r - g).mean())
    E.log(f"  [validate] descriptions max|diff|={maxd:.2e} mean={meand:.2e} | "
          f"cosine mean={cos.mean():.5f} min={cos.min():.4f}")
    ok = float(cos.mean()) > 0.999
    E.log(f"  descriptor {arch} [{preset}] [{mode}]: {'OK' if ok else 'MISMATCH'}")
    return ("descriptor_" + arch, preset, side, n, ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=["fast", "balanced"],
                    choices=list(PRESETS))
    ap.add_argument("--archs", nargs="+", default=["dedode_b"],
                    choices=["dedode_b", "dedode_g"])
    ap.add_argument("--components", nargs="+", default=["detector", "descriptor"],
                    choices=["detector", "descriptor"])
    ap.add_argument("--outdir", default="onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--im", default="assets/0015_B.jpg")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    # Detector weights are identical for every variant; build the lightest once.
    if "detector" in args.components:
        det_model = E.build_model("B128")
        for p in args.presets:
            try:
                results.append(export_detector_preset(det_model, p, args.outdir,
                                                       args.opset, args.im))
            except Exception as e:
                results.append(("detector", p, PRESETS[p]["det"],
                                PRESETS[p]["kpts"], False))
                E.log(f"  detector [{p}] FAILED: {type(e).__name__}: {e}")
        del det_model

    if "descriptor" in args.components:
        for arch in args.archs:
            model = E.build_model(ARCH_VARIANT[arch])
            for p in args.presets:
                try:
                    results.append(export_descriptor_preset(model, arch, p,
                                                             args.outdir, args.opset, args.im))
                except Exception as e:
                    results.append(("descriptor_" + arch, p,
                                    PRESETS[p]["desc"][arch], PRESETS[p]["kpts"], False))
                    E.log(f"  descriptor {arch} [{p}] FAILED: {type(e).__name__}: {e}")
            del model

    E.log("==================== PRESET SUMMARY ====================")
    for comp, p, side, n, ok in results:
        E.log(f"  {comp:20s} {p:9s} {side}px {n}kpts  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
