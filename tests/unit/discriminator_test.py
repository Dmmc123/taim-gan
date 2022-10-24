from src.models.modules import Discriminator

import torch

import pytest


@pytest.mark.parametrize(
    argnames=("batch_num", "height", "width", "num_words"),
    argvalues=(
        (64, 14, 14, 1),
        (32, 7,  2,  3),
        (16, 3,  7,  7),
    )
)
def test_d(batch_num, height, width, num_words):
    # generate fake rgb images
    v = torch.rand((batch_num, 3, height, width))
    # generate contextual embeddings of the same channels
    w = torch.rand((batch_num, 3, num_words))
    D = Discriminator(emb_len=3)
    logits = D(v, w)
    assert logits.size() == (batch_num, num_words), "output dims should never change"
