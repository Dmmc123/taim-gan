from src.models.modules import Discriminator

import torch

import pytest


@pytest.mark.parametrize(
    argnames=("batch_num", "chans", "height", "width", "num_words"),
    argvalues=(
        (64, 3,  14, 14, 1),
        (32, 2,  20, 20, 2),
        (16, 1,  17, 14, 4),
        (8,  1,  32, 32, 8),
        (4,  5,  16, 16, 16),
        (2,  10, 13, 18, 32),
        (1,  7,  13, 26, 64),
    )
)
def test_d(batch_num, chans, height, width, num_words):
    v = torch.rand((batch_num, chans, height, width))
    w = torch.rand((batch_num, chans, num_words))
    D = Discriminator(img_chans=chans)
    logits = D(v, w)
    assert logits.size(0) == batch_num and logits.size(1) == num_words, "output dims should never change"
