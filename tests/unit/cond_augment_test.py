from src.models.modules import CondAugmentation

import torch

import pytest

@pytest.mark.parametrize(
    argnames=("batch", "D", "conditioning_dim"),
    argvalues=(
            (16, 128, 64),
            (32, 256, 128),
            (64, 64, 64),
    )
)

def test_cond_augment(batch, D, conditioning_dim):
    cond_augment = CondAugmentation(D, conditioning_dim)
    text_embedding = torch.randn(batch, D)
    c_hat, mu, logvar = cond_augment(text_embedding)
    assert c_hat.shape == (batch, conditioning_dim), "c_hat dimension is not matching conditioning_dim."
    assert mu.shape == (batch, conditioning_dim), "mu dimension is not matching conditioning_dim."
    assert logvar.shape == (batch, conditioning_dim), "logvar dimension is not matching conditioning_dim."