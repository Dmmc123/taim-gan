"""UpSample module."""

from torch import nn


def up_sample(in_planes: int, out_planes: int) -> nn.Module:
    """UpSample module."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(
            in_planes, out_planes * 2, kernel_size=3, stride=1, padding=1, bias=False
        ),
        nn.InstanceNorm2d(out_planes * 2),
        nn.GLU(dim=1),
    )


def img_up_block(in_planes: int, out_planes: int) -> nn.Module:
    """Image upsample block."""
    return nn.Sequential(
        nn.Upsample(scale_factor=1.9, mode="nearest"),
        nn.Conv2d(
            in_planes, out_planes * 2, kernel_size=3, stride=1, padding=1, bias=False
        ),
        nn.InstanceNorm2d(out_planes * 2),
        nn.GLU(dim=1),
    )
