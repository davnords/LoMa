"""LoMa ONNX-Runtime-CUDA benchmark sweep + charts.

Sweeps detector resolution / keypoint budget for the B128 (DeDoDe-B) pipeline,
exporting any missing detector/descriptor ONNX on demand, benchmarks each config on
GPU via the CUDA execution provider, validates against PyTorch, writes a JSON report
and renders publication-quality charts for the README.

    uv run bench_sweep.py            # export (if needed) + benchmark + chart
    uv run bench_sweep.py --no-bench # only (re)render charts from benchmark.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import export_onnx as E
from export_jetson import preprocess_square

IM_A, IM_B = "assets/0015_A.jpg", "assets/0015_B.jpg"
ONNX_DIR = "onnx/bench"
DOCS = "docs"
OPSET = 18
ITERS, WARMUP = 30, 12

# name, detector (H, W), descriptor side, keypoints
CONFIGS = [
    dict(name="pico",      det=(256, 256),  desc=256, kpts=512),   # fastest (Jetson RT)
    dict(name="nano",      det=(256, 256),  desc=256, kpts=1024),  # ~4x faster than fast
    dict(name="turbo",     det=(384, 384),  desc=384, kpts=1024),  # ~2x faster than fast
    dict(name="turbo-2k",  det=(384, 384),  desc=384, kpts=2048),  # fast + many matches
    dict(name="fast",      det=(512, 512),  desc=512, kpts=1024),
    dict(name="fast-2k",   det=(512, 512),  desc=512, kpts=2048),
    dict(name="balanced",  det=(640, 640),  desc=640, kpts=1536),
    dict(name="wide",      det=(512, 1024), desc=512, kpts=2048),
    dict(name="quality",   det=(1024, 1024), desc=784, kpts=2048),
]

PALETTE = dict(detect="#4C72B0", describe="#DD8452", match="#55A868",
               accent="#C44E52", grid="#E6E6E6", fg="#2B2B2B")


# --------------------------------------------------------------------------- #
# export on demand
# --------------------------------------------------------------------------- #
def det_path(H, W, k):
    return f"{ONNX_DIR}/det_{H}x{W}_{k}.onnx"


def desc_path(side):
    return f"{ONNX_DIR}/desc_b_{side}.onnx"


def ensure_detector(model, H, W, k):
    p = det_path(H, W, k)
    if os.path.exists(p):
        return p
    E.log(f"export detector {H}x{W} k={k}")
    model._detector.subpixel = True
    img = preprocess_square(IM_A, (H, W), E.dev(model))
    w = E.prepare(E.DetectorWrapper(model, k))
    E.do_export(w, (img,), p, ["image"], ["keypoints", "keypoint_probs"],
                OPSET, dynamic_shapes=None, allow_legacy=False)
    return p


def ensure_descriptor(model, side):
    from torch.export import Dim
    p = desc_path(side)
    if os.path.exists(p):
        return p
    E.log(f"export descriptor {side}x{side}")
    img = preprocess_square(IM_A, side, E.dev(model))
    g = torch.Generator().manual_seed(0)
    kp = (2 * torch.rand(1, 2048, 2, generator=g) - 1).float().to(E.dev(model))
    w = E.prepare(E.DescriptorWrapper(model))
    b, nk = Dim("batch"), Dim("nkpts", min=16)
    E.do_export(w, (img, kp), p, ["image", "keypoints"], ["descriptions"],
                OPSET, dynamic_shapes=({0: b}, {0: b, 1: nk}))
    return p


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #
def gpu_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        path, sess_options=so,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])


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


def ransac_inliers(P, thresh=3.0):
    """Geometric-verification inliers via a fundamental matrix (USAC_MAGSAC),
    matching LoMa's own RANSAC usage. Returns (n_inliers, inlier_ratio)."""
    import cv2
    if len(P) < 8:
        return 0, 0.0
    _, mask = cv2.findFundamentalMat(
        np.ascontiguousarray(P[:, :2]), np.ascontiguousarray(P[:, 2:]),
        cv2.USAC_MAGSAC, thresh, 0.999999, 10000)
    n = int(mask.sum()) if mask is not None else 0
    return n, n / max(len(P), 1)


def run_benchmark():
    import onnxruntime as ort
    from PIL import Image
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "no CUDA EP"
    os.makedirs(ONNX_DIR, exist_ok=True)
    wA, hA = Image.open(IM_A).size
    wB, hB = Image.open(IM_B).size

    m = E.build_model("B128")
    d = E.dev(m)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    detW = E.prepare(E.DetectorWrapper(m, 1024))
    descW = E.prepare(E.DescriptorWrapper(m))
    matchW = E.prepare(E.MatcherWrapper(m, m.cfg.filter_threshold))
    mat_s = gpu_session("onnx/loma_matcher_B128.onnx")

    rows = []
    for c in CONFIGS:
        H, W = c["det"]; side = c["desc"]; k = c["kpts"]
        dp = ensure_detector(m, H, W, k)
        sp = ensure_descriptor(m, side)
        det_s, dsc_s = gpu_session(dp), gpu_session(sp)
        m._detector.subpixel = True; detW.num_keypoints = k

        na = E.to_np(preprocess_square(IM_A, (H, W), d))
        nb = E.to_np(preprocess_square(IM_B, (H, W), d))
        nda = E.to_np(preprocess_square(IM_A, side, d))
        ndb = E.to_np(preprocess_square(IM_B, side, d))
        kA, _ = det_s.run(None, {"image": na}); kB, _ = det_s.run(None, {"image": nb})
        dA = dsc_s.run(None, {"image": nda, "keypoints": kA})[0]
        dB = dsc_s.run(None, {"image": ndb, "keypoints": kB})[0]
        feed = {"kpts0": kA, "kpts1": kB, "desc0": dA, "desc1": dB}
        mo = mat_s.run(None, feed)

        t_det = timed(lambda: det_s.run(None, {"image": na}))
        t_dsc = timed(lambda: dsc_s.run(None, {"image": nda, "keypoints": kA}))
        t_mat = timed(lambda: mat_s.run(None, feed))
        total = 2 * t_det + 2 * t_dsc + t_mat

        with torch.no_grad():
            ka, _ = detW(preprocess_square(IM_A, (H, W), d))
            kb, _ = detW(preprocess_square(IM_B, (H, W), d))
            da = descW(preprocess_square(IM_A, side, d), ka)
            db = descW(preprocess_square(IM_B, side, d), kb)
            mt = matchW(ka, kb, da, db)
        P_pt = pairs(E.to_np(mt[0]), E.to_np(ka), E.to_np(kb), hA, wA, hB, wB)
        P_on = pairs(mo[0], kA, kB, hA, wA, hB, wB)

        inl, ratio = ransac_inliers(P_on)
        row = dict(name=c["name"], H=H, W=W, desc=side, kpts=k,
                   mpix=round(H * W / 1e6, 3),
                   detect_ms=round(2 * t_det, 2), describe_ms=round(2 * t_dsc, 2),
                   match_ms=round(t_mat, 2), total_ms=round(total, 2),
                   fps=round(1000.0 / total, 2), matches=len(P_on),
                   inliers=inl, inlier_ratio=round(ratio * 100, 1),
                   fidelity=round(overlap(P_on, P_pt, 2.0) * 100, 2))
        rows.append(row)
        E.log(f"{c['name']:9} {H}x{W} k={k}: total {row['total_ms']}ms "
              f"{row['fps']}fps {row['matches']}m {inl}inl ({row['inlier_ratio']}%) "
              f"{row['fidelity']}%")

    os.makedirs(DOCS, exist_ok=True)
    meta = dict(gpu=gpu, runtime=f"onnxruntime {ort.__version__} (CUDA EP)",
                variant="B128 / DeDoDe-B", pair=f"{os.path.basename(IM_A)}/{os.path.basename(IM_B)}",
                rows=rows)
    with open(f"{DOCS}/benchmark.json", "w") as f:
        json.dump(meta, f, indent=2)
    E.log(f"wrote {DOCS}/benchmark.json")
    return meta


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25, "font.size": 11.5,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#9aa0a6", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": PALETTE["grid"], "grid.linewidth": 0.9,
        "axes.axisbelow": True, "text.color": PALETTE["fg"],
        "axes.labelcolor": PALETTE["fg"], 
        "xtick.color": PALETTE["fg"], "ytick.color": PALETTE["fg"],
        "figure.facecolor": "white", "axes.facecolor": "#FBFBFC",
        "legend.frameon": False,
    })
    return plt


def _titles(ax, main, sub):
    """Clean two-line heading: bold title + grey subtitle, no overlap."""
    ax.set_title(sub, fontsize=9.5, color="#7a7f87", pad=8)
    ax.annotate(main, xy=(0.0, 1.0), xytext=(0.0, 22), xycoords="axes fraction",
                textcoords="offset points", fontsize=14, fontweight="bold",
                color=PALETTE["fg"], ha="left", va="bottom", annotation_clip=False)


def make_charts(meta):
    plt = _style()
    rows = meta["rows"]
    names = [r["name"] for r in rows]
    detect = [r["detect_ms"] for r in rows]
    describe = [r["describe_ms"] for r in rows]
    match = [r["match_ms"] for r in rows]
    total = [r["total_ms"] for r in rows]
    fps = [r["fps"] for r in rows]
    matches = [r["matches"] for r in rows]
    x = np.arange(len(rows))
    sub = f"{meta['variant']}   ·   {meta['gpu']}   ·   {meta['runtime']}"

    def chart_latency(ax):
        ax.bar(x, detect, label="detect ×2", color=PALETTE["detect"], width=0.62)
        ax.bar(x, describe, bottom=detect, label="describe ×2", color=PALETTE["describe"], width=0.62)
        ax.bar(x, match, bottom=np.add(detect, describe), label="match", color=PALETTE["match"], width=0.62)
        for i, t in enumerate(total):
            ax.text(i, t + max(total) * 0.02, f"{t:.0f} ms\n{fps[i]:.1f} fps",
                    ha="center", va="bottom", fontsize=8.5, color=PALETTE["fg"], linespacing=1.25)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("latency per match() call (ms)")
        ax.set_ylim(0, max(total) * 1.22)
        ax.legend(ncol=3, loc="upper left", fontsize=9.5, columnspacing=1.1)
        _titles(ax, "ONNX-CUDA latency breakdown", sub)

    def chart_pareto(ax):
        sc = ax.scatter(total, matches, s=170, c=fps, cmap="viridis",
                        edgecolors="white", linewidths=1.6, zorder=3)
        for r in rows:
            dx, dy = (10, 6)
            if r["name"] == "wide":
                dx, dy = (10, -16)
            ax.annotate(r["name"], (r["total_ms"], r["matches"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=10.5)
        cb = ax.figure.colorbar(sc, ax=ax, pad=0.02); cb.set_label("FPS", fontsize=10)
        cb.outline.set_visible(False)
        ax.set_xlabel("latency per match() call (ms)   →   slower")
        ax.set_ylabel("# matches   →   more")
        ax.margins(0.12)
        _titles(ax, "Speed vs accuracy (Pareto front)", sub)

    def chart_fps(ax):
        bars = ax.bar(x, fps, color=PALETTE["accent"], width=0.6)
        for b, v in zip(bars, fps):
            ax.text(b.get_x() + b.get_width() / 2, v + max(fps) * 0.025, f"{v:.1f}",
                    ha="center", fontsize=9.5)
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("throughput (pairs / s)")
        ax.set_ylim(0, max(fps) * 1.18)
        _titles(ax, "Throughput by preset", sub)

    def chart_scaling(ax):
        # detect scales with detector pixels; describe with descriptor pixels.
        def dedup(key_px, val):
            agg = {}
            for r, v in zip(rows, val):
                agg.setdefault(r[key_px], []).append(v)
            xs = sorted(agg)
            return xs, [float(np.mean(agg[k])) for k in xs]
        det_mp = [(r["H"] * r["W"]) / 1e6 for r in rows]
        dsc_mp = [(r["desc"] ** 2) / 1e6 for r in rows]
        # attach mpix keys
        for r, a, b in zip(rows, det_mp, dsc_mp):
            r["_detmp"], r["_dscmp"] = round(a, 3), round(b, 3)
        dx, dy = dedup("_detmp", detect); ex, ey = dedup("_dscmp", describe)
        ax.plot(dx, dy, "o-", label="detect ×2  (vs detector px)", color=PALETTE["detect"], lw=2.4, ms=8)
        ax.plot(ex, ey, "s-", label="describe ×2  (vs descriptor px)", color=PALETTE["describe"], lw=2.4, ms=8)
        ax.set_xlabel("input resolution (megapixels)"); ax.set_ylabel("latency (ms)")
        ax.legend(loc="upper left", fontsize=9.5)
        ax.margins(x=0.08, y=0.18)
        _titles(ax, "Latency scales ~linearly with pixels", sub)

    def chart_inliers(ax):
        # matches vs RANSAC-verified inliers (fundamental matrix, 3 px) per preset
        inliers = [r.get("inliers", 0) for r in rows]
        ratio = [r.get("inlier_ratio", 0) for r in rows]
        w = 0.4
        ax.bar(x - w / 2, matches, w, label="matches", color="#B9C6DD")
        b2 = ax.bar(x + w / 2, inliers, w, label="RANSAC inliers (F, 3 px)", color=PALETTE["match"])
        for i, bb in enumerate(b2):
            ax.text(bb.get_x() + bb.get_width() / 2, inliers[i] + max(matches) * 0.02,
                    f"{inliers[i]}\n{ratio[i]:.0f}%", ha="center", va="bottom",
                    fontsize=8, color=PALETTE["fg"], linespacing=1.2)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("correspondences")
        ax.set_ylim(0, max(matches) * 1.2)
        ax.legend(loc="upper left", fontsize=9.5)
        _titles(ax, "Geometric inliers per preset", sub)

    for name, fn, size in [
        ("latency_breakdown", chart_latency, (8.2, 4.8)),
        ("speed_accuracy", chart_pareto, (8.0, 5.2)),
        ("fps", chart_fps, (8.2, 4.2)),
        ("latency_scaling", chart_scaling, (8.0, 4.8)),
        ("inliers", chart_inliers, (8.6, 4.6)),
    ]:
        fig, ax = plt.subplots(figsize=size)
        fn(ax)
        fig.savefig(f"{DOCS}/{name}.png"); plt.close(fig)

    # combined 2×3 dashboard (README hero)
    fig, axs = plt.subplots(2, 3, figsize=(22, 10))
    chart_latency(axs[0, 0]); chart_pareto(axs[0, 1]); chart_inliers(axs[0, 2])
    chart_fps(axs[1, 0]); chart_scaling(axs[1, 1])
    axs[1, 2].axis("off")
    fig.suptitle("LoMa · ONNX Runtime CUDA benchmark", fontsize=18, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4.5, w_pad=3.0)
    fig.savefig(f"{DOCS}/dashboard.png"); plt.close(fig)

    E.log(f"wrote 6 charts (incl. dashboard) to {DOCS}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bench", action="store_true", help="only re-render charts")
    args = ap.parse_args()
    if args.no_bench:
        meta = json.load(open(f"{DOCS}/benchmark.json"))
    else:
        meta = run_benchmark()
    make_charts(meta)


if __name__ == "__main__":
    main()
