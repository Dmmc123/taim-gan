from src.models.modules import up_sample, img_up_block

import torch

import pytest

@pytest.mark.parametrize(
    argnames=("batch", "in_channel", "out_channel", "height", "width"),
    argvalues=(
            (16, 32, 64, 32, 32),
            (32, 16, 32, 64, 64),
            (64, 8, 16, 128, 128),
    )
)

def test_upsample(batch, in_channel, out_channel, height, width):
    upsample = up_sample(in_channel, out_channel)
    input = torch.randn(batch, in_channel, height, width)
    out = upsample(input)
    assert out.size(0) == input.size(0) and out.size(1) == out_channel and out.size(2) == height * 2 and out.size(3) == width * 2, "upsampled dimension is incorrect, should be 2x of input dimension."

@pytest.mark.parametrize(
    argnames=("batch", "in_channel", "out_channel", "height", "width"),
    argvalues=(
            (16, 32, 64, 32, 32),
            (32, 16, 32, 64, 64),
            (64, 8, 16, 128, 128),
    )
)

def test_img_up_block(batch, in_channel, out_channel, height, width):
    imgUpBlock = img_up_block(in_channel, out_channel)
    input = torch.randn(batch, in_channel, height, width)
    out = imgUpBlock(input)
    assert out.size(0) == input.size(0) and out.size(1) == out_channel and out.size(2) == int(height * 1.9) and out.size(3) == int(width * 1.9), "upsampled dimension is incorrect, should be 1.9x of input dimension."