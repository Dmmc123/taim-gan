from src.models.modules import InceptionEncoder, VGGEncoder

import torch

import pytest

@pytest.mark.parametrize(
    argnames=("D", "batch"),
    argvalues=(
            (16, 2),
            (8, 4),
            (4, 8),
    )
)
def test_inception_image_encoder(D, batch, channel = 3, height = 256, width = 256):
    encoder = InceptionEncoder(D)
    input = torch.randn(batch, channel, height, width)
    local_img_features, global_img_features = encoder(input)
    assert local_img_features.size(1) == D and (local_img_features.size(2) == local_img_features.size(3) == 17), "local img feature dimension is wrong."
    assert global_img_features.size(1) == D, "global img feature dimension is wrong."


@pytest.mark.parametrize(
    argnames="batch",
    argvalues=(1, 2, 3)
)
def test_vgg_1(batch):
    """check forward pass dim output"""
    vgg = VGGEncoder()
    x = torch.randn(batch, 3, 256, 256)
    output = vgg(x)
    assert output.size() == (batch, 128, 128, 128), "vgg output dimension is wrong."


def test_vgg_2():
    """check edge-case for returning None"""
    vgg = VGGEncoder()
    vgg.select = "some nonexistent layer"
    x = torch.randn(2, 3, 256, 256)
    output = vgg(x)
    assert output is None, "if there's no layers of interest, vgg should return None"
