import torch


def imagenet(img: torch.Tensor) -> torch.Tensor:
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    return (img - imagenet_mean[None, :, None, None]) / imagenet_std[
        None, :, None, None
    ]


def inception(img: torch.Tensor) -> torch.Tensor:
    inception_mean = torch.tensor([0.5, 0.5, 0.5], device=img.device)
    inception_std = torch.tensor([0.5, 0.5, 0.5], device=img.device)
    return (img - inception_mean[None, :, None, None]) / inception_std[
        None, :, None, None
    ]
