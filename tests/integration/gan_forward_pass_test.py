from src.models.modules import (
    Discriminator, Generator,
    TextEncoder, VGGEncoder, InceptionEncoder
)

import torch

import pytest

IMG_CHANS = 3  # RGB channels for image
IMG_HW = 256  # height and width of images
HIDDEN_DIM = 128  # hidden dimensions of lstm cell in one direction
C = 2 * HIDDEN_DIM  # length of embeddings


@pytest.mark.parametrize(
    argnames=("Ng", "cond_dim", "z_dim", "vocab_size", "word_emb_dim", "batch_size", "L"),
    argvalues=(
        (16, 10, 10, 10,  100, 2, 100),
        (32, 20, 5,  100, 25,  4, 50),
        (64, 5,  17, 200, 77,  6, 17),
    )
)
def test_gan(Ng, cond_dim, z_dim, vocab_size, word_emb_dim, batch_size, L):
    """
    :param int Ng: depth of feature maps
    :param int cond_dim: length of conditioning vector
    :param int z_dim: length of a noise vector
    :param int vocab_size: amount of unique words in all textual input
    :param int word_emb_dim: length of embeddings for each word
    :param int batch_size: amount of samples in one training batch
    :param int L: amount of words in one input string
    """
    # define D, G, and visual/textual encoders
    D = Discriminator(n_words=L)
    G = Generator(Ng=Ng, D=C, conditioning_dim=cond_dim, noise_dim=z_dim)
    lstm = TextEncoder(vocab_size=vocab_size, emb_dim=word_emb_dim, hidden_dim=HIDDEN_DIM)
    vgg = VGGEncoder()
    inception = InceptionEncoder(D=C)

    # generate some noise
    noise = torch.rand((batch_size, z_dim))
    # generate inout for textual encoders and get word and sentence embeddings
    tokens = torch.randint(0, vocab_size, (batch_size, L))
    word_embs, sent_embs = lstm(tokens)
    # generate visual features from vgg and inception
    images = torch.rand((batch_size, IMG_CHANS, IMG_HW, IMG_HW))
    vgg_features = vgg(images)
    local_features, global_features = inception(images)
    # obtain fake images
    fake_images, _, _ = G(noise, sent_embs, word_embs, global_features, local_features, vgg_features)
    # propagate fake images through discriminator and get logits
    logits_word_level, logits_uncond, logits_cond = D(fake_images, word_embs)


