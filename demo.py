from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw
from loma import LoMa
from loma.loma import LoMaB, LoMaConfig, filter_matches, to_pixel_coords


def main(
    cfg: LoMaConfig = LoMaB(),
    im_A: str = "assets/toronto_A.jpg",
    im_B: str = "assets/toronto_B.jpg",
    save_path: str = "demo/matches.jpg",
):
    model = LoMa(cfg)

    # NOTE: you could simply call kptsA, kptsB = model.match(im_A, im_B)
    # Here we unpack the internals to see what's happening:
    kpts_A, desc_A, h1, w1 = model.detect_and_describe(im_A)
    kpts_B, desc_B, h2, w2 = model.detect_and_describe(im_B)
    with torch.inference_mode():
        scores = model(kpts_A, kpts_B, desc_A, desc_B)["scores"]
    m0, *_ = filter_matches(scores, model.cfg.filter_threshold)
    valid = m0[0] > -1
    matched_A = to_pixel_coords(kpts_A[0][torch.where(valid)[0]], h1, w1).cpu().numpy()
    matched_B = to_pixel_coords(kpts_B[0][m0[0][valid]], h2, w2).cpu().numpy()

    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)))
    canvas.paste(Image.open(im_A).convert("RGB"), (0, 0))
    canvas.paste(Image.open(im_B).convert("RGB"), (w1, 0))
    draw = ImageDraw.Draw(canvas)
    rng = np.random.default_rng(0)
    for (x1, y1), (x2, y2) in zip(matched_A, matched_B):
        color = tuple(rng.integers(0, 256, 3).tolist())
        draw.line([(x1, y1), (x2 + w1, y2)], fill=color, width=1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    print(f"Saved {len(matched_A)} matches to {save_path}")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.CascadeSubcommandArgs,))
