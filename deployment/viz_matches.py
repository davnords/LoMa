"""Render a qualitative LoMa-ONNX matching figure for the README.

Runs the ONNX pipeline (detector -> descriptor -> matcher) on an image pair and draws
the correspondences side-by-side. Saves docs/matches.png.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

import export_onnx as E
from export_jetson import preprocess_square

IM_A, IM_B = "assets/0015_A.jpg", "assets/0015_B.jpg"
PRESET = dict(det=512, desc=512, kpts=2048)  # fast-2k: lots of matches
MAX_DRAW = 220


def sess(p):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"])
    return ort.InferenceSession(p, sess_options=so, providers=prov)


def main():
    import torch
    m = E.build_model("B128"); d = E.dev(m)
    det = sess("onnx/loma_detector_fast.onnx" if PRESET["kpts"] == 1024
               else "onnx/bench/det_512x512_2048.onnx")
    dsc = sess("onnx/loma_descriptor_dedode_b_fast.onnx")
    mat = sess("onnx/loma_matcher_B128.onnx")

    na = E.to_np(preprocess_square(IM_A, PRESET["det"], d))
    nb = E.to_np(preprocess_square(IM_B, PRESET["det"], d))
    sa = E.to_np(preprocess_square(IM_A, PRESET["desc"], d))
    sb = E.to_np(preprocess_square(IM_B, PRESET["desc"], d))
    kA, _ = det.run(None, {"image": na}); kB, _ = det.run(None, {"image": nb})
    dA = dsc.run(None, {"image": sa, "keypoints": kA})[0]
    dB = dsc.run(None, {"image": sb, "keypoints": kB})[0]
    m0 = mat.run(None, {"kpts0": kA, "kpts1": kB, "desc0": dA, "desc1": dB})[0][0]

    A = Image.open(IM_A).convert("RGB"); B = Image.open(IM_B).convert("RGB")
    wA, hA = A.size; wB, hB = B.size
    H = max(hA, hB)
    canvas = Image.new("RGB", (wA + wB, H), (12, 12, 16))
    canvas.paste(A, (0, 0)); canvas.paste(B, (wA, 0))
    draw = ImageDraw.Draw(canvas)

    valid = np.where(m0 > -1)[0]
    rng = np.random.default_rng(0)
    if len(valid) > MAX_DRAW:
        valid = rng.choice(valid, MAX_DRAW, replace=False)
    n_total = int((m0 > -1).sum())

    def px(k, w, h):
        return (w * (k[0] + 1) / 2, h * (k[1] + 1) / 2)

    for i in valid:
        j = int(m0[i])
        ax, ay = px(kA[0][i], wA, hA)
        bx, by = px(kB[0][j], wB, hB)
        c = tuple(int(v) for v in rng.integers(60, 256, 3))
        draw.line([(ax, ay), (bx + wA, by)], fill=c, width=2)
        draw.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], outline=c, width=2)
        draw.ellipse([bx + wA - 3, by - 3, bx + wA + 3, by + 3], outline=c, width=2)

    os.makedirs("docs", exist_ok=True)
    # downscale for a tidy README asset
    scale = 1600 / canvas.width
    out = canvas.resize((1600, int(canvas.height * scale)))
    out.save("docs/matches.png")
    print(f"wrote docs/matches.png — drew {len(valid)} of {n_total} matches")


if __name__ == "__main__":
    main()
