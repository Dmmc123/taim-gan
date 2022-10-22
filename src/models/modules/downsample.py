"""downsample module."""

from torch import nn


def down_sample(in_planes: int, out_planes: int) -> nn.Module:
    """UpSample module."""
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )
