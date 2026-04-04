from time import perf_counter

from loma import LoMa
import torch
from loma.device import device


def test_throughput():
    model = LoMa(LoMa.Cfg(compile=True)).to(device)
    img_A = torch.randn(1, 3, 784, 784).to(device)
    num_kpts = 2048
    # warmup
    for i in range(10):
        model._detector(img_A, num_kpts)
    # measure throughput
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = perf_counter()
    T = 20
    for i in range(T):
        model._detector(img_A, num_kpts)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = perf_counter()
    print(f"Throughput: {T / (end_time - start_time)} fps")


if __name__ == "__main__":
    test_throughput()
