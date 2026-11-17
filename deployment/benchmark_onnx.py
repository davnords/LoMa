"""Benchmark the LoMa ONNX presets: latency + quality + size.

For each preset (fast/balanced/quality) of the B128 pipeline it reports:
  * model sizes (MB)
  * ONNX latency per stage (detector/descriptor/matcher) via onnxruntime
  * PyTorch GPU latency of the same stages (reference; this dev box has a fast GPU,
    Jetson numbers come from the C++ tool cpp/examples/benchmark.cpp)
  * quality: #matches, and match-pair overlap of the ONNX pipeline vs the PyTorch
    pipeline at the same resolution.

Absolute latency here is NOT the Jetson number -- run loma_bench on the Orin Nano
for that. This shows the relative cost/quality trade-off between presets and confirms
ONNX==PyTorch per preset.
"""

import os
import time

import numpy as np
import torch

import export_onnx as E
from export_jetson import PRESETS, preprocess_square

IM_A, IM_B = "assets/0015_A.jpg", "assets/0015_B.jpg"
ARCH = "dedode_b"
VARIANT = "B128"
DESC_DIM = 128
ITERS = 30
WARMUP = 5


def mb(path):
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else float("nan")


def to_pixel(norm, h, w):
    out = norm.copy()
    out[:, 0] = w * (norm[:, 0] + 1) / 2
    out[:, 1] = h * (norm[:, 1] + 1) / 2
    return out


def match_pairs(m0, kA, kB, hA, wA, hB, wB):
    m0 = m0[0]
    v = m0 > -1
    iA = np.where(v)[0]
    iB = m0[v]
    pA = to_pixel(kA[0][iA], hA, wA)
    pB = to_pixel(kB[0][iB], hB, wB)
    return np.concatenate([pA, pB], 1)


def overlap(P, Q, tol=2.0):
    if len(P) == 0 or len(Q) == 0:
        return 0.0
    d = np.sqrt(((P[:, None] - Q[None]) ** 2).reshape(len(P), len(Q), 2, 2).sum(-1))
    return float(((d[..., 0] < tol) & (d[..., 1] < tol)).any(1).mean())


def timed(fn, iters=ITERS, warmup=WARMUP, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    from PIL import Image
    wA, hA = Image.open(IM_A).size
    wB, hB = Image.open(IM_B).size

    m = E.build_model(VARIANT)
    d = E.dev(m)
    detW = E.prepare(E.DetectorWrapper(m, PRESETS["fast"]["kpts"]))  # kpts reset per preset below
    descW = E.prepare(E.DescriptorWrapper(m))
    matchW = E.prepare(E.MatcherWrapper(m, m.cfg.filter_threshold))
    matcher_path = f"onnx/loma_matcher_{VARIANT}.onnx"
    mat_s = E.ort_session(matcher_path)

    print(f"\n{'='*92}\nLoMa ONNX benchmark — variant {VARIANT} ({ARCH}), image pair {os.path.basename(IM_A)}/{os.path.basename(IM_B)}")
    print(f"ONNX latency = onnxruntime CPU on this box | PyTorch latency = {d} (reference, NOT Jetson)")
    print(f"{'='*92}")
    hdr = (f"{'preset':9} {'det':>4} {'desc':>4} {'kpts':>5} | {'sizeMB':>7} | "
           f"{'ONNX det':>9} {'ONNX dsc':>9} {'ONNX mat':>9} {'ONNX tot':>9} {'FPS':>6} | "
           f"{'pt-GPU tot':>10} | {'#match':>6} {'vs-PT@2px':>9}")
    print(hdr)
    print("-" * len(hdr))

    for preset, cfg in PRESETS.items():
        det_side = cfg["det"]; desc_side = cfg["desc"][ARCH]; n = cfg["kpts"]
        det_p = f"onnx/loma_detector_{preset}.onnx"
        dsc_p = f"onnx/loma_descriptor_{ARCH}_{preset}.onnx"
        if not (os.path.exists(det_p) and os.path.exists(dsc_p)):
            print(f"{preset:9} (missing onnx: {det_p} / {dsc_p}) — skipped")
            continue
        det_s = E.ort_session(det_p)
        dsc_s = E.ort_session(dsc_p)
        m._detector.subpixel = True

        det_a = preprocess_square(IM_A, det_side, d)
        det_b = preprocess_square(IM_B, det_side, d)
        dsc_a = preprocess_square(IM_A, desc_side, d)
        dsc_b = preprocess_square(IM_B, desc_side, d)
        na = E.to_np(det_a); nb = E.to_np(det_b)
        nda = E.to_np(dsc_a); ndb = E.to_np(dsc_b)

        # ---- ONNX latency (CPU) ----
        kA, _ = det_s.run(None, {"image": na})
        kB, _ = det_s.run(None, {"image": nb})
        t_det = timed(lambda: det_s.run(None, {"image": na}))
        dA = dsc_s.run(None, {"image": nda, "keypoints": kA})[0]
        dB = dsc_s.run(None, {"image": ndb, "keypoints": kB})[0]
        t_dsc = timed(lambda: dsc_s.run(None, {"image": nda, "keypoints": kA}))
        feed = {"kpts0": kA, "kpts1": kB, "desc0": dA, "desc1": dB}
        mo = mat_s.run(None, feed)
        t_mat = timed(lambda: mat_s.run(None, feed))
        t_tot = 2 * t_det + 2 * t_dsc + t_mat
        fps = 1000.0 / t_tot

        # ---- PyTorch GPU latency (reference) ----
        nkp = n
        detW.num_keypoints = nkp
        with torch.no_grad():
            def pt_pipeline():
                ka, _ = detW(det_a); kb, _ = detW(det_b)
                da = descW(dsc_a, ka); db = descW(dsc_b, kb)
                return matchW(ka, kb, da, db)
            t_pt = timed(pt_pipeline, iters=15, warmup=3, cuda=(d.type == "cuda"))
            # reference matches (PyTorch)
            ka, _ = detW(det_a); kb, _ = detW(det_b)
            da = descW(dsc_a, ka); db = descW(dsc_b, kb)
            mt = matchW(ka, kb, da, db)
        P_pt = match_pairs(E.to_np(mt[0]), E.to_np(ka), E.to_np(kb), hA, wA, hB, wB)

        # ---- quality (ONNX pipeline matches vs PyTorch) ----
        P_on = match_pairs(mo[0], kA, kB, hA, wA, hB, wB)
        ov = overlap(P_on, P_pt, tol=2.0)

        size = mb(det_p) + mb(dsc_p) + mb(matcher_path)
        print(f"{preset:9} {det_side:>4} {desc_side:>4} {n:>5} | {size:>7.1f} | "
              f"{t_det:>8.2f}m {t_dsc:>8.2f}m {t_mat:>8.2f}m {t_tot:>8.2f}m {fps:>6.1f} | "
              f"{t_pt:>9.2f}m | {len(P_on):>6} {ov*100:>8.1f}%")

    print(f"\nNotes: ONNX latency is CPU (this box). For Jetson Orin Nano GPU numbers, "
          f"build cpp/ and run loma_bench. 'vs-PT@2px' = fraction of ONNX matches within "
          f"2px of a PyTorch match (same preset).")


if __name__ == "__main__":
    main()
