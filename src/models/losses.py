"""Module containing the loss functions for the GANs."""
from typing import Any

import torch
from torch import nn


def discriminator_loss(
    logits: dict[str, dict[str, torch.Tensor]],
    true_labels: torch.Tensor,
    fake_labels: torch.Tensor,
    lambda_4: float = 1.0,
) -> Any:
    """
    Calculate discriminator objective

    :param dict[str, dict[str, torch.Tensor]] logits:
        Dictionary with fake/real and word-level/uncond/cond logits
    :param torch.Tensor true_labels: True labels
    :param torch.Tensor fake_labels: Fake labels
    :param float lambda_4: Hyperparameter for word loss in paper
    :return: Discriminator objective loss
    :rtype: Any
    """
    # define main loss functions for logit losses
    bce_logits = nn.BCEWithLogitsLoss()
    bce = nn.BCELoss()
    # calculate word-level loss
    word_loss = bce(logits["real"]["word_level"], true_labels)
    word_loss += bce(logits["fake"]["word_level"], fake_labels)
    # calculate unconditional adversarial loss
    uncond_loss = bce_logits(logits["real"]["uncond"], true_labels)
    uncond_loss += bce_logits(logits["fake"]["uncond"], fake_labels)
    # calculate conditional adversarial loss
    cond_loss = bce_logits(logits["real"]["cond"], true_labels)
    cond_loss += bce_logits(logits["fake"]["cond"], fake_labels)
    return -1 / 2 * (uncond_loss + cond_loss) + lambda_4 * word_loss
