from time import perf_counter
import tyro
from PIL import Image
from tqdm import tqdm

import torch
import numpy as np

from loma.device import device
from loma import LoMa, LoMaB
from loma.cfg import LoMaConfig

def test_throughput(matcher: LoMaConfig = LoMaB()):
    model = LoMa(matcher)
    im_A = Image.open("assets/toronto_A.jpg").resize((784, 784))
    im_B = Image.open("assets/toronto_B.jpg").resize((784, 784))
    im_A = (
        torch.from_numpy(np.array(im_A)).permute(2, 0, 1).unsqueeze(0).to(device) / 255
    )
    im_B = (
        torch.from_numpy(np.array(im_B)).permute(2, 0, 1).unsqueeze(0).to(device) / 255
    )
    # warmup
    for i in tqdm(range(10), desc="Warming up..."):
        model.match(im_A, im_B)
    # measure throughput
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = perf_counter()
    T = 20
    for i in tqdm(range(T), desc="Measuring throughput..."):
        model.match(im_A, im_B)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = perf_counter()
    print(f"Throughput: {T / (end_time - start_time)} fps")


if __name__ == "__main__":
    tyro.cli(test_throughput, config=(tyro.conf.CascadeSubcommandArgs,))
