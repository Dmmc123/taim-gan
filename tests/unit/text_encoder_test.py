from src.models.modules import TextEncoder

import torch

import pytest


@pytest.mark.parametrize(
    argnames=("emb_dim", "hidden_dim", "batch", "vocab_len", "n_words"),
    argvalues=(
        (50,  25,  64, 500,  100),
        (100, 50,  32, 3000, 120),
        (150, 100, 1,  500,  200),
        (30,  200, 16, 5000, 100),
        (123, 300, 8,  100,  30),
    )
)
def test_text_encoder(emb_dim, hidden_dim, batch, vocab_len, n_words):
    encoder = TextEncoder(vocab_len, emb_dim, hidden_dim)
    tokens = torch.randint(0, n_words, (batch, n_words))
    w, s = encoder(tokens)
    assert w.size() == (batch, hidden_dim, n_words), "word embeddings should be BxCxL"
    assert s.size() == (batch, hidden_dim), "sentence embeddings should be BxC"

