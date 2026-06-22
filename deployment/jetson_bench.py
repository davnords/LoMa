"""Standalone LoMa-ONNX benchmark for Jetson — onnxruntime + numpy + PIL only.

No PyTorch / loma / opencv needed. Runs the B128 pipeline (detector -> descriptor ->
matcher) at the requested presets on the Jetson GPU (CUDA/TensorRT EP) or CPU, and
reports per-stage latency, FPS and match count.

    python3 jetson_bench.py --onnx onnx --imgA 0015_A.jpg --imgB 0015_B.jpg \
            --presets pico nano turbo fast --iters 20
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

# name -> (detH, detW, descSide, kpts)
PRESETS = {
    "pico":     (256, 256, 256, 512),
    "nano":     (256, 256, 256, 1024),
    "turbo":    (384, 384, 384, 1024),
    "fast":     (512, 512, 512, 1024),
    "balanced": (640, 640, 640, 1536),
    "wide":     (512, 1024, 512, 2048),
    "quality":  (1024, 1024, 784, 2048),
}


def providers():
    av = ort.get_available_providers()
    chain = []
    # TensorRT EP needs libnvinfer (not installed here) -> stick to CUDA -> CPU.
    if "CUDAExecutionProvider" in av:
        chain.append(("CUDAExecutionProvider", {"device_id": 0}))
    chain.append("CPUExecutionProvider")
    return chain


def make_session(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        return ort.InferenceSession(path, sess_options=so, providers=providers())
    except Exception:
        # CUDA/TensorRT EP failed to initialize (e.g. Tegra mismatch) -> CPU.
        return ort.InferenceSession(path, sess_options=so,
                                    providers=["CPUExecutionProvider"])


def out_idx(sess, name):
    for i, o in enumerate(sess.get_outputs()):
        if o.name == name:
            return i
    return 0


def load_chw(path, H, W):
    im = Image.open(path).convert("RGB").resize((W, H))
    a = (np.asarray(im) / 255.0).astype(np.float32)
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1))[None])  # [1,3,H,W]


def to_pixel(n, h, w):
    o = n.copy()
    o[:, 0] = w * (n[:, 0] + 1) / 2
    o[:, 1] = h * (n[:, 1] + 1) / 2
    return o


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="onnx")
    ap.add_argument("--imgA", default="0015_A.jpg")
    ap.add_argument("--imgB", default="0015_B.jpg")
    ap.add_argument("--presets", nargs="+",
                    default=["pico", "nano", "turbo", "fast"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", default="jetson_bench.json")
    args = ap.parse_args()

    print("onnxruntime", ort.__version__, "| available providers:",
          ort.get_available_providers())
    matcher = make_session(os.path.join(args.onnx, "loma_matcher_B128.onnx"))
    used = matcher.get_providers()[0]
    print("active provider:", used)
    wA, hA = Image.open(args.imgA).size
    wB, hB = Image.open(args.imgB).size
    m0_i = out_idx(matcher, "m0")

    hdr = (f"{'preset':9}{'det':>10}{'desc':>6}{'kpts':>6} | "
           f"{'det_ms':>8}{'dsc_ms':>8}{'mat_ms':>8}{'total':>9}{'FPS':>7}{'#match':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for p in args.presets:
        if p not in PRESETS:
            print(f"{p}: unknown preset"); continue
        H, W, D, K = PRESETS[p]
        dp = os.path.join(args.onnx, f"loma_detector_{p}.onnx")
        sp = os.path.join(args.onnx, f"loma_descriptor_dedode_b_{p}.onnx")
        if not (os.path.exists(dp) and os.path.exists(sp)):
            print(f"{p:9} missing onnx ({os.path.basename(dp)} / {os.path.basename(sp)})")
            continue
        det = make_session(dp)
        dsc = make_session(sp)
        kp_i = out_idx(det, "keypoints")

        na, nb = load_chw(args.imgA, H, W), load_chw(args.imgB, H, W)
        nda, ndb = load_chw(args.imgA, D, D), load_chw(args.imgB, D, D)

        kA = det.run(None, {"image": na})[kp_i]
        kB = det.run(None, {"image": nb})[kp_i]
        dA = dsc.run(None, {"image": nda, "keypoints": kA})[0]
        dB = dsc.run(None, {"image": ndb, "keypoints": kB})[0]
        feed = {"kpts0": kA, "kpts1": kB, "desc0": dA, "desc1": dB}
        mo = matcher.run(None, feed)
        m0 = mo[m0_i][0]
        nmatch = int((m0 > -1).sum())

        t_det = timed(lambda: det.run(None, {"image": na}), args.iters, args.warmup)
        t_dsc = timed(lambda: dsc.run(None, {"image": nda, "keypoints": kA}),
                      args.iters, args.warmup)
        t_mat = timed(lambda: matcher.run(None, feed), args.iters, args.warmup)
        total = 2 * t_det + 2 * t_dsc + t_mat
        fps = 1000.0 / total
        print(f"{p:9}{H}x{W:<6}{D:>6}{K:>6} | "
              f"{t_det:>7.2f}m{t_dsc:>7.2f}m{t_mat:>7.2f}m{total:>8.1f}m{fps:>7.2f}{nmatch:>7}")
        rows.append(dict(preset=p, H=H, W=W, desc=D, kpts=K,
                         det_ms=round(t_det, 2), dsc_ms=round(t_dsc, 2),
                         mat_ms=round(t_mat, 2), total_ms=round(total, 1),
                         fps=round(fps, 2), matches=nmatch))
        del det, dsc  # free GPU memory between presets (Orin Nano 8GB)

    with open(args.out, "w") as f:
        json.dump(dict(provider=used, runtime=ort.__version__, rows=rows), f, indent=2)
    print(f"\nwrote {args.out}  | provider={used}")


if __name__ == "__main__":
    main()
