"""GPU benchmark of the LoMa pipeline across presets (PyTorch CUDA).

On the DGX Spark / GB10 the Python `onnxruntime` package has no aarch64 GPU wheel,
so the available Python GPU runtime is PyTorch CUDA. The exported ONNX models
reproduce these PyTorch ops bit-for-bit (proven: cosine=1.0, 99.5% match overlap),
so this is the representative GPU cost. For true ONNX-Runtime-CUDA numbers, build
cpp/ against a CUDA ORT (see README) and run loma_bench on GB10 / Orin Nano.

Measures per-stage GPU latency (detector / descriptor / matcher), throughput and
#matches for each preset of the B128 (DeDoDe-B) pipeline.
"""

import os
import time

import numpy as np
import torch

import export_onnx as E
from export_jetson import PRESETS, preprocess_square

IM_A, IM_B = "assets/0015_A.jpg", "assets/0015_B.jpg"
ARCH, VARIANT, DESC_DIM = "dedode_b", "B128", 128
ITERS, WARMUP = 50, 10


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    for _ in range(WARMUP):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    sync()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def main():
    dev = E.dev_str() if hasattr(E, "dev_str") else None
    m = E.build_model(VARIANT)
    d = E.dev(m)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    detW = E.prepare(E.DetectorWrapper(m, PRESETS["fast"]["kpts"]))
    descW = E.prepare(E.DescriptorWrapper(m))
    matchW = E.prepare(E.MatcherWrapper(m, m.cfg.filter_threshold))

    print(f"\n{'='*100}")
    print(f"LoMa GPU benchmark — {VARIANT} ({ARCH}) on {gpu}  [{d}]  | pair {os.path.basename(IM_A)}/{os.path.basename(IM_B)}")
    print(f"{'='*100}")
    hdr = (f"{'preset':9} {'det':>4} {'desc':>4} {'kpts':>5} | "
           f"{'detect(2x)':>10} {'describe(2x)':>12} {'match':>8} {'TOTAL':>8} {'FPS':>6} | {'#match':>6}")
    print(hdr)
    print("-" * len(hdr))

    for preset, cfg in PRESETS.items():
        det_side, desc_side, n = cfg["det"], cfg["desc"][ARCH], cfg["kpts"]
        m._detector.subpixel = True
        detW.num_keypoints = n
        da_img = preprocess_square(IM_A, det_side, d)
        db_img = preprocess_square(IM_B, det_side, d)
        sa_img = preprocess_square(IM_A, desc_side, d)
        sb_img = preprocess_square(IM_B, desc_side, d)

        with torch.no_grad():
            # warmup keypoints for descriptor/matcher timing
            kA, _ = detW(da_img); kB, _ = detW(db_img)
            dA = descW(sa_img, kA); dB = descW(sb_img, kB)

            t_det = timed(lambda: detW(da_img)) + timed(lambda: detW(db_img))
            t_dsc = timed(lambda: descW(sa_img, kA)) + timed(lambda: descW(sb_img, kB))
            t_mat = timed(lambda: matchW(kA, kB, dA, dB))
            t_tot = t_det + t_dsc + t_mat

            mt = matchW(kA, kB, dA, dB)
            nm = int((E.to_np(mt[0])[0] > -1).sum())

        fps = 1000.0 / t_tot
        print(f"{preset:9} {det_side:>4} {desc_side:>4} {n:>5} | "
              f"{t_det:>9.2f}m {t_dsc:>11.2f}m {t_mat:>7.2f}m {t_tot:>7.2f}m {fps:>6.1f} | {nm:>6}")

    print(f"\nNotes:")
    print(f"  * detect/describe columns are for BOTH images (the cost of one match() call).")
    print(f"  * GB10 is far faster than Orin Nano 8GB; treat these as relative trends.")
    print(f"  * For on-device ONNX-Runtime-CUDA latency, run cpp/loma_bench on the target.")


if __name__ == "__main__":
    main()
