import torch
from torch import nn


class RandomCrop1D(nn.Module):
    def __init__(self, crop_samples: int):
        super().__init__()
        assert crop_samples > 0
        self.crop_samples = crop_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        t = x.shape[-1]
        target = self.crop_samples

        if t < target:
            n_repeat = (target + t - 1) // t
            x = x.repeat(1, n_repeat)
            t = x.shape[-1]

        if t == target:
            out = x
        else:
            start = int(torch.randint(0, t - target + 1, (1,)).item())
            out = x[..., start : start + target]

        if squeeze:
            out = out.squeeze(0)
        return out
