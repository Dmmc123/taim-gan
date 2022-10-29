from src.models.modules import Generator

import torch

import pytest


@pytest.mark.parametrize(
    argnames=("batch", "Ng", "D", "conditioning_dim", "noise_dim", "L"),
    argvalues=(
            (2,  32, 16, 100, 100, 18),
            (4,  32, 32, 20,  50,  12),
            (2,  32, 15, 50,  200, 14),
    )
)
def test_generator(batch, Ng, D, conditioning_dim, noise_dim, L):
    generator = Generator(Ng, D, conditioning_dim, noise_dim)
    noise = torch.randn(batch, noise_dim)
    sentence_embeddings = torch.randn(batch, D)
    word_embeddings = torch.randn(batch, D, L)
    global_inception_feat = torch.randn(batch, D)
    local_inception_feat = torch.randn(batch, D, 17, 17)
    vgg_feat = torch.randn(batch, D // 2, 128, 128)
    
    fake_image, mu_tensor, logvar = generator(noise, sentence_embeddings, word_embeddings, global_inception_feat, local_inception_feat, vgg_feat)
    
    assert fake_image.shape == (batch, 3, 256, 256), "fake_image dimension is not matching (batch, 3, 256, 256)."
    assert mu_tensor.shape == (batch, conditioning_dim), "mu dimension is not matching conditioning_dim."
    assert logvar.shape == (batch, conditioning_dim), "logvar dimension is not matching conditioning_dim."
