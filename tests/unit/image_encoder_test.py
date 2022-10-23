from src.models.modules import InceptionEncoder, VGGEncoder

import torch

import pytest

@pytest.mark.parametrize(
    argnames=("D", "batch"),
    argvalues=(
            (256, 16),
            (128, 32),
            (64, 64),
    )
)
def test_inception_image_encoder(D, batch, channel = 3, height = 256, width = 256):
    encoder = InceptionEncoder(D)
    input = torch.randn(batch, channel, height, width)
    local_img_features, global_img_features = encoder(input)
    assert local_img_features.size(1) == D and (local_img_features.size(2) == local_img_features.size(3) == 17), "local img feature dimension is wrong."
    assert global_img_features.size(1) == D, "global img feature dimension is wrong."

@pytest.mark.parametrize(
    argnames=("batch"),
    argvalues=(
            (2),
            (4),
            (8),
    )
)
def test_vgg(batch):
    vgg = VGGEncoder()
    input = torch.randn(batch, 3, 256, 256)
    output = vgg(input)
    assert output.size(1) == output.size(2) == output.size(3) == 128, "vgg output dimension is wrong."
