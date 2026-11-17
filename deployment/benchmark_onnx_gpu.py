"""Benchmark + validate the LoMa ONNX pipeline on GPU via onnxruntime CUDA EP.

Requires an onnxruntime build with CUDAExecutionProvider (built from source on the
DGX Spark GB10 — see build_ort_gpu.sh). Reports per-stage GPU latency, throughput,
#matches, and ONNX-CUDA vs PyTorch match-pair overlap, for each preset.
"""

import os
import time

import numpy as np
import onnxruntime as ort
import torch

import export_onnx as E
from export_jetson import PRESETS, preprocess_square, as_hw

IM_A, IM_B = "assets/0015_A.jpg", "assets/0015_B.jpg"
ARCH, VARIANT, DESC_DIM = "dedode_b", "B128", 128
ITERS, WARMUP = 30, 12


def gpu_session(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def timed(fn):
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def to_pixel(norm, h, w):
    o = norm.copy(); o[:, 0] = w * (norm[:, 0] + 1) / 2; o[:, 1] = h * (norm[:, 1] + 1) / 2
    return o


def pairs(m0, kA, kB, hA, wA, hB, wB):
    m0 = m0[0]; v = m0 > -1
    return np.concatenate([to_pixel(kA[0][np.where(v)[0]], hA, wA),
                           to_pixel(kB[0][m0[v]], hB, wB)], 1)


def overlap(P, Q, tol=2.0):
    if len(P) == 0 or len(Q) == 0:
        return 0.0
    d = np.sqrt(((P[:, None] - Q[None]) ** 2).reshape(len(P), len(Q), 2, 2).sum(-1))
    return float(((d[..., 0] < tol) & (d[..., 1] < tol)).any(1).mean())


def main():
    from PIL import Image
    wA, hA = Image.open(IM_A).size
    wB, hB = Image.open(IM_B).size

    print("onnxruntime", ort.__version__, "| providers:", ort.get_available_providers())
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "no CUDA EP!"

    m = E.build_model(VARIANT)
    d = E.dev(m)
    detW = E.prepare(E.DetectorWrapper(m, PRESETS["fast"]["kpts"]))
    descW = E.prepare(E.DescriptorWrapper(m))
    matchW = E.prepare(E.MatcherWrapper(m, m.cfg.filter_threshold))
    mat_s = gpu_session(f"onnx/loma_matcher_{VARIANT}.onnx")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"

    print(f"\n{'='*104}")
    print(f"LoMa ONNX-Runtime-CUDA benchmark — {VARIANT} ({ARCH}) on {gpu}  | pair {os.path.basename(IM_A)}/{os.path.basename(IM_B)}")
    print(f"{'='*104}")
    hdr = (f"{'preset':9} {'det':>9} {'desc':>4} {'kpts':>5} | "
           f"{'ONNX det':>9} {'ONNX dsc':>9} {'ONNX mat':>9} {'TOTAL(2x)':>10} {'FPS':>6} | "
           f"{'#match':>6} {'vs-PT@2px':>9}")
    print(hdr); print("-" * len(hdr))

    for preset, cfg in PRESETS.items():
        det_side, desc_side, n = cfg["det"], cfg["desc"][ARCH], cfg["kpts"]
        dh, dw = as_hw(det_side)
        det_lbl = f"{dh}x{dw}" if dh != dw else str(dh)
        det_p = f"onnx/loma_detector_{preset}.onnx"
        dsc_p = f"onnx/loma_descriptor_{ARCH}_{preset}.onnx"
        if not (os.path.exists(det_p) and os.path.exists(dsc_p)):
            print(f"{preset:9} (missing onnx) — skipped"); continue
        det_s = gpu_session(det_p); dsc_s = gpu_session(dsc_p)
        m._detector.subpixel = True; detW.num_keypoints = n

        na = E.to_np(preprocess_square(IM_A, det_side, d))
        nb = E.to_np(preprocess_square(IM_B, det_side, d))
        nda = E.to_np(preprocess_square(IM_A, desc_side, d))
        ndb = E.to_np(preprocess_square(IM_B, desc_side, d))

        kA, _ = det_s.run(None, {"image": na}); kB, _ = det_s.run(None, {"image": nb})
        dA = dsc_s.run(None, {"image": nda, "keypoints": kA})[0]
        dB = dsc_s.run(None, {"image": ndb, "keypoints": kB})[0]
        feed = {"kpts0": kA, "kpts1": kB, "desc0": dA, "desc1": dB}
        mo = mat_s.run(None, feed)

        t_det = timed(lambda: det_s.run(None, {"image": na}))
        t_dsc = timed(lambda: dsc_s.run(None, {"image": nda, "keypoints": kA}))
        t_mat = timed(lambda: mat_s.run(None, feed))
        t_tot = 2 * t_det + 2 * t_dsc + t_mat
        fps = 1000.0 / t_tot

        # quality vs PyTorch (same preset)
        with torch.no_grad():
            ka, _ = detW(preprocess_square(IM_A, det_side, d))
            kb, _ = detW(preprocess_square(IM_B, det_side, d))
            da = descW(preprocess_square(IM_A, desc_side, d), ka)
            db = descW(preprocess_square(IM_B, desc_side, d), kb)
            mt = matchW(ka, kb, da, db)
        P_pt = pairs(E.to_np(mt[0]), E.to_np(ka), E.to_np(kb), hA, wA, hB, wB)
        P_on = pairs(mo[0], kA, kB, hA, wA, hB, wB)
        ov = overlap(P_on, P_pt, 2.0)

        print(f"{preset:9} {det_lbl:>9} {desc_side:>4} {n:>5} | "
              f"{t_det:>8.2f}m {t_dsc:>8.2f}m {t_mat:>8.2f}m {t_tot:>9.2f}m {fps:>6.1f} | "
              f"{len(P_on):>6} {ov*100:>8.1f}%")

    print("\n* TOTAL(2x) = 2x detector + 2x descriptor + 1x matcher (cost of one match()).")
    print("* vs-PT@2px = fraction of ONNX-CUDA matches within 2px of a PyTorch match.")


if __name__ == "__main__":
    main()
