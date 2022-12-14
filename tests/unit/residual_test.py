from src.models.modules import ResidualBlock

import torch

import pytest

@pytest.mark.parametrize(
    argnames=("batch", "channel_num", "height", "width"),
    argvalues=(
            # big dims do oom error on runner, upgrade costs money
            (2, 128, 128, 128),
            (1, 32, 64, 64),
            (4, 2, 64, 64),
    )
)
def test_residual(batch, channel_num, height, width):
    residual = ResidualBlock(channel_num)
    input = torch.randn(batch, channel_num, height, width)
    out = residual(input)
    assert out.size(0) == input.size(0) and out.size(1) == input.size(1) and out.size(2) == input.size(2) and out.size(3) == input.size(3), "output dimension is not matching input dimension."