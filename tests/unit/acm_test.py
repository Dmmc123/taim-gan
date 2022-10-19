from src.models.modules import ACM
import torch

import pytest

@pytest.mark.parametrize(
    argnames=("text_chans", "img_chans", "inner_dim", "batch_dim", "height", "width"),
    argvalues=(
        (3,  10, 2,  64,  10, 10),
        (1,  1,  4,  128, 1, 1),
        (10, 10, 5,  128, 16, 1),
        (6,  1,  8,  128, 100, 100),
        (1,  9,  16, 64,  2, 2),
    )
)
def test_vanilla_acm(text_chans, img_chans, inner_dim, batch_dim, height, width):
    acm = ACM(
        img_chans=img_chans,
        text_chans=text_chans,
        inner_dim=inner_dim
    )
    text = torch.rand((batch_dim, text_chans, height, width))
    img = torch.rand((batch_dim, img_chans, height, width))
    # check that propagation doesn't raise errors
    res = acm(text, img)
    return True
