from src.models.modules import ACM
import torch


def test_vanilla_acm():
    acm = ACM(
        text_chans=3,
        img_chans=10,
        inner_dim=15
    )
    text = torch.rand((64, 3, 128, 64))
    img = torch.rand((64, 10, 132, 68))
    res = acm(text, img)
    return True
