from time import perf_counter

import torch

from loma import LoMa
from loma.device import device


def test_throughput():
    model = LoMa(LoMa.Cfg(compile=True)).to(device)
    img_A = torch.randn(1, 3, 784, 784).to(device)
    num_kpts = 2048
    keypoints = torch.empty(1, num_kpts, 2, device=device).uniform_(-1.0, 1.0)

    for _ in range(10):
        model._descriptor.describe_keypoints(img_A, keypoints)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = perf_counter()
    T = 20
    for _ in range(T):
        model._descriptor.describe_keypoints(img_A, keypoints)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = perf_counter()
    print(f"Throughput: {T / (end_time - start_time)} fps")


if __name__ == "__main__":
    test_throughput()
