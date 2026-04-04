from time import perf_counter

from torch.cpu import is_available
from loma import LoMa
import torch
from PIL import Image
import numpy as np
from loma.device import device

def test_throughput():
    # speeds things up a little bit
    torch.backends.cudnn.deterministic = False
    
    # device = torch.device("cuda")
    model = LoMa(LoMa.Cfg(compile=True)).to(device)
    im_A = Image.open("assets/toronto_A.jpg").resize((560, 560))
    im_B = Image.open("assets/toronto_B.jpg").resize((560, 560))
    im_A = torch.from_numpy(np.array(im_A)).permute(2, 0, 1).unsqueeze(0).to(device) / 255
    im_B = torch.from_numpy(np.array(im_B)).permute(2, 0, 1).unsqueeze(0).to(device) / 255
    # warmup
    for i in range(10):
        model.match(im_A, im_B)
    # measure throughput
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = perf_counter()
    T = 20
    for i in range(T):
        model.match(im_A, im_B)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = perf_counter()
    print(f"Throughput: {T / (end_time - start_time)} fps")

if __name__ == "__main__":
    test_throughput()