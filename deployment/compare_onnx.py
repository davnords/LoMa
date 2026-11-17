"""Compare the B128 ONNX pipeline against the original PyTorch code.

Two comparisons on the same image pair:
  (1) Per-stage graph fidelity  -- identical inputs into PyTorch vs ONNX for each
      stage (detector / descriptor / matcher) in isolation.
  (2) Full end-to-end           -- ONNX det->desc->match  vs  PyTorch det->desc->match,
      comparing the final matched pixel-coordinate pairs.

Both pipelines use identical preprocessing (detector @752x1024, descriptor @784x784)
so any difference is the ONNX<->PyTorch gap, not preprocessing.
"""

import numpy as np
import torch
from PIL import Image

import export_onnx as E

IM_A, IM_B = "assets/0015_A.jpg", "assets/0015_B.jpg"
DET_H, DET_W = 752, 1024
DESC_HW = 784
N = 2048


def preprocess(path, H, W, device):
    im = Image.open(path).convert("RGB").resize((W, H))
    arr = np.array(im) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).float()[None].to(device)


def to_pixel(norm_xy, h, w):
    out = norm_xy.copy()
    out[:, 0] = w * (norm_xy[:, 0] + 1) / 2
    out[:, 1] = h * (norm_xy[:, 1] + 1) / 2
    return out


def matches_pixels(m0, kA, kB, hA, wA, hB, wB):
    m0 = m0[0]
    valid = m0 > -1
    iA = np.where(valid)[0]
    iB = m0[valid]
    pA = to_pixel(kA[0][iA], hA, wA)
    pB = to_pixel(kB[0][iB], hB, wB)
    return np.concatenate([pA, pB], axis=1)  # [M,4] = (xA,yA,xB,yB)


def pair_overlap(P, Q, tol=2.0):
    """Fraction of P pairs that have a Q pair within tol px on BOTH endpoints."""
    if len(P) == 0 or len(Q) == 0:
        return 0.0
    d = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).reshape(len(P), len(Q), 2, 2).sum(-1))
    matched = ((d[..., 0] < tol) & (d[..., 1] < tol)).any(1)
    return float(matched.mean())


def main():
    torch.manual_seed(0)
    m = E.build_model("B128")
    d = E.dev(m)

    detW = E.prepare(E.DetectorWrapper(m, N))
    descW = E.prepare(E.DescriptorWrapper(m))
    matchW = E.prepare(E.MatcherWrapper(m, m.cfg.filter_threshold))

    det_a = preprocess(IM_A, DET_H, DET_W, d)
    det_b = preprocess(IM_B, DET_H, DET_W, d)
    dsc_a = preprocess(IM_A, DESC_HW, DESC_HW, d)
    dsc_b = preprocess(IM_B, DESC_HW, DESC_HW, d)

    det_s = E.ort_session("onnx/loma_detector.onnx")
    dsc_s = E.ort_session("onnx/loma_descriptor_dedode_b.onnx")
    mat_s = E.ort_session("onnx/loma_matcher_B128.onnx")

    np_ = lambda t: t.detach().cpu().numpy()

    print("=" * 70)
    print("(1) PER-STAGE GRAPH FIDELITY (identical inputs, PyTorch vs ONNX)")
    print("=" * 70)

    # --- detector ---
    with torch.no_grad():
        kA_t, pA_t = detW(det_a)
    kA_o, pA_o = det_s.run(None, {"image": np_(det_a)})
    det_agree = E.kpt_agreement(np_(kA_t), kA_o, tol=1e-3)
    print(f"detector   : keypoint-set agreement {det_agree*100:.2f}% | "
          f"probs max|diff| {np.abs(np_(pA_t)-pA_o).max():.2e}")

    # --- descriptor: feed the SAME (PyTorch) keypoints to both ---
    with torch.no_grad():
        dA_t = descW(dsc_a, kA_t)
    dA_o = dsc_s.run(None, {"image": np_(dsc_a), "keypoints": np_(kA_t)})[0]
    print(f"descriptor : descriptions max|diff| {np.abs(np_(dA_t)-dA_o).max():.2e} "
          f"mean {np.abs(np_(dA_t)-dA_o).mean():.2e}")

    # --- matcher: feed SAME kpts+descs (from PyTorch detector/descriptor) to both ---
    with torch.no_grad():
        kB_t, _ = detW(det_b)
        dB_t = descW(dsc_b, kB_t)
        mt = matchW(kA_t, kB_t, dA_t, dB_t)
    mo = mat_s.run(None, {"kpts0": np_(kA_t), "kpts1": np_(kB_t),
                          "desc0": np_(dA_t), "desc1": np_(dB_t)})
    m0_agree = float((np_(mt[0]) == mo[0]).mean())
    npy = int((np_(mt[0])[0] > -1).sum())
    non = int((mo[0][0] > -1).sum())
    print(f"matcher    : m0 index agreement {m0_agree*100:.2f}% | "
          f"#matches PyTorch={npy} ONNX={non}")

    print()
    print("=" * 70)
    print("(2) FULL END-TO-END  (ONNX det->desc->match  vs  PyTorch det->desc->match)")
    print("=" * 70)
    wA, hA = Image.open(IM_A).size
    wB, hB = Image.open(IM_B).size

    # PyTorch pipeline
    with torch.no_grad():
        kAt, _ = detW(det_a); kBt, _ = detW(det_b)
        dAt = descW(dsc_a, kAt); dBt = descW(dsc_b, kBt)
        mt = matchW(kAt, kBt, dAt, dBt)
    P_py = matches_pixels(np_(mt[0]), np_(kAt), np_(kBt), hA, wA, hB, wB)

    # ONNX pipeline
    kAo, _ = det_s.run(None, {"image": np_(det_a)})
    kBo, _ = det_s.run(None, {"image": np_(det_b)})
    dAo = dsc_s.run(None, {"image": np_(dsc_a), "keypoints": kAo})[0]
    dBo = dsc_s.run(None, {"image": np_(dsc_b), "keypoints": kBo})[0]
    mo = mat_s.run(None, {"kpts0": kAo, "kpts1": kBo, "desc0": dAo, "desc1": dBo})
    P_on = matches_pixels(mo[0], kAo, kBo, hA, wA, hB, wB)

    print(f"# final matches : PyTorch={len(P_py)}  ONNX={len(P_on)}")
    for tol in (1.0, 2.0, 5.0):
        f_po = pair_overlap(P_py, P_on, tol)
        f_op = pair_overlap(P_on, P_py, tol)
        print(f"  match-pair overlap @ {tol:>3.0f}px : "
              f"PyTorch->ONNX {f_po*100:5.1f}% | ONNX->PyTorch {f_op*100:5.1f}%")

    # also report the library's own match() as a sanity reference (aspect-ratio preproc)
    a, b = m.match(IM_A, IM_B)
    print(f"\n(reference) library model.match() returns {len(a)} matches "
          f"(uses aspect-ratio preprocessing, so count differs from fixed-size pipeline)")


if __name__ == "__main__":
    main()
