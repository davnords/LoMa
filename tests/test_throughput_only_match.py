from time import perf_counter

from loma import LoMa
import torch
from loma.device import device


def test_throughput():
    model = LoMa(LoMa.Cfg(compile=True)).to(device)
    kpts_A = torch.randn(1, 2048, 2).to(device)
    kpts_B = torch.randn(1, 2048, 2).to(device)
    feats_A = torch.randn(1, 2048, 256).to(device)
    feats_B = torch.randn(1, 2048, 256).to(device)

    # warmup
    for i in range(10):
        model(kpts_A, kpts_B, feats_A, feats_B)
    # measure throughput
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = perf_counter()
    T = 20
    for i in range(T):
        model(kpts_A, kpts_B, feats_A, feats_B)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = perf_counter()
    print(f"Throughput: {T / (end_time - start_time)} fps")


if __name__ == "__main__":
    test_throughput()
