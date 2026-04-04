import torch

from loma import LoMa
from loma.descriptor.dedode import DeDoDeDescriptor
from loma.detector.dad import DaD


def test_public_api_imports() -> None:
    assert LoMa is not None


def test_detector_dense_probs() -> None:
    detector = DaD(DaD.Cfg(compile=False))
    images = torch.rand(1, 3, 64, 64)

    result = detector.detect(images, num_keypoints=32, return_dense_probs=True)

    assert "dense_probs" in result
    assert result["keypoints"].shape == (1, 32, 2)
    assert result["keypoint_probs"].shape == (1, 32)
    assert result["dense_probs"].shape == (1, 64, 64)


def test_descriptor_forward() -> None:
    descriptor = DeDoDeDescriptor(DeDoDeDescriptor.Cfg(compile=False))
    images = torch.rand(1, 3, 64, 64)
    keypoints = torch.empty(1, 16, 2).uniform_(-1.0, 1.0)

    result = descriptor.describe_keypoints(images, keypoints)

    assert result["descriptions"].shape == (1, 16, 256)


if __name__ == "__main__":
    test_public_api_imports()
    test_detector_dense_probs()
    test_descriptor_forward()
