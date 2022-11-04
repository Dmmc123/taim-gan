from src.models.modules import Discriminator

import torch

import pytest

RGB = 3
IMG_HW = 256
C = IMG_HW


@pytest.mark.parametrize(
    argnames=("batch_num", "num_words"),
    argvalues=(
        (16, 1),
        (4,  3),
        (1,  7),
    )
)
def test_d(batch_num, num_words):
    v = torch.rand((batch_num, RGB, IMG_HW, IMG_HW))
    # generate contextual embeddings of the same channels
    w = torch.rand((batch_num, C, num_words))
    s = torch.rand((batch_num, C))
    D = Discriminator()
    img_feat = D(v)
    logits_word_level = D.logits_word_level(img_feat, w)
    logits_uncond = D.logits_uncond(img_feat)
    logits_cond = D.logits_cond(img_feat, s)
    assert logits_word_level.size() == (batch_num, num_words), "for word logits dims should be BxL"
    assert logits_uncond.size() == logits_cond.size() == (batch_num, 1), "for real/fake logits dims should be Bx1"
