"""Module containing the loss functions for the GANs."""
from typing import Any

import torch
from torch import nn

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


def generator_loss(
    logits: dict[str, dict[str, torch.Tensor]],
    local_fake_incept_feat: torch.Tensor,
    global_fake_incept_feat: torch.Tensor,
    real_labels: torch.Tensor,
    word_labels: torch.Tensor,
    words_emb: torch.Tensor,
    sent_emb: torch.Tensor,
    match_labels: torch.Tensor,
    cap_lens: torch.Tensor,
    class_ids: torch.Tensor,
    real_vgg_feat: torch.Tensor,
    fake_vgg_feat: torch.Tensor,
    const_dict: dict[str, float],
) -> Any:
    """Calculate the loss for the generator.

    Args:
        logits: Dictionary with fake/real and word-level/uncond/cond logits

        local_fake_incept_feat: The local inception features for the fake images.

        global_fake_incept_feat: The global inception features for the fake images.

        real_labels: Label for "real" image as predicted by discriminator,
        this is a tensor of ones. [shape: (batch_size, 1)].

        word_labels: POS tagged word labels for the captions. [shape: (batch_size, L)]

        words_emb: The embeddings for all the words in the captions.
        shape: (batch_size, embedding_size, max_caption_length)

        sent_emb: The embeddings for the sentences.
        shape: (batch_size, embedding_size)

        match_labels: Tensor of shape: (batch_size, 1).
        This is of the form torch.tensor([0, 1, 2, ..., batch-1])

        cap_lens: The length of the 'actual' captions in the batch [without padding]
        shape: (batch_size, 1)

        class_ids: The class ids for the instance. shape: (batch_size, 1)

        real_vgg_feat: The vgg features for the real images. shape: (batch_size, 128, 128, 128)
        fake_vgg_feat: The vgg features for the fake images. shape: (batch_size, 128, 128, 128)

        const_dict: The dictionary containing the constants.
    """
    lambda1 = const_dict["lambda1"]
    lambda2 = const_dict["lambda2"]
    total_error_g = 0.0

    cond_logits = logits["fake"]["cond"]
    cond_err_g = nn.BCEWithLogitsLoss()(cond_logits, real_labels)

    uncond_logits = logits["fake"]["uncond"]
    uncond_err_g = nn.BCEWithLogitsLoss()(uncond_logits, real_labels)

    # add up the conditional and unconditional losses
    loss_g = -0.5 * (cond_err_g + uncond_err_g)
    total_error_g += loss_g

    # DAMSM Loss from attnGAN.
    loss_damsm = damsm_loss(
        local_fake_incept_feat,
        global_fake_incept_feat,
        words_emb,
        sent_emb,
        match_labels,
        cap_lens,
        class_ids,
        const_dict,
    )

    total_error_g += loss_damsm

    loss_per = nn.MSELoss()(real_vgg_feat, fake_vgg_feat)  # perceptual loss

    total_error_g += lambda1 * loss_per

    word_level_loss = nn.BCELoss()(logits["fake"]["word_level"], word_labels)
    total_error_g += lambda2 * word_level_loss

    return total_error_g


def damsm_loss(
    local_incept_feat: torch.Tensor,
    global_incept_feat: torch.Tensor,
    words_emb: torch.Tensor,
    sent_emb: torch.Tensor,
    match_labels: torch.Tensor,
    cap_lens: torch.Tensor,
    class_ids: torch.Tensor,
    const_dict: dict[str, float],
) -> Any:
    """Calculate the DAMSM loss from the attnGAN paper.

    Args:
        local_incept_feat: The local inception features. [shape: (batch, D, 17, 17)]

        global_incept_feat: The global inception features. [shape: (batch, D)]

        words_emb: The embeddings for all the words in the captions.

        shape: (batch, D, max_caption_length)

        sent_emb: The embeddings for the sentences. shape: (batch_size, D)

        match_labels: Tensor of shape: (batch_size, 1).
        This is of the form torch.tensor([0, 1, 2, ..., batch-1])

        cap_lens: The length of the 'actual' captions in the batch [without padding]
        shape: (batch_size, 1)

        class_ids: The class ids for the instance. shape: (batch, 1)

        const_dict: The dictionary containing the constants.
    """
    batch_size = match_labels.size(0)
    # Mask mis-match samples, that come from the same class as the real sample
    masks = []

    match_scores = []
    gamma1 = const_dict["gamma1"]
    gamma2 = const_dict["gamma2"]
    gamma3 = const_dict["gamma3"]
    lambda3 = const_dict["lambda3"]

    for i in range(batch_size):
        mask = (class_ids == class_ids[i]).int()
        # This ensures that "correct class" index is not included in the mask.
        mask[i] = 0
        masks.append(mask.reshape(1, -1))  # shape: (1, batch)

        numb_words = int(cap_lens[i])
        # shape: (1, D, L), this picks the caption at ith batch index.
        query_words = words_emb[i, :, :numb_words].unsqueeze(0)
        # shape: (batch, D, L), this expands the same caption for all batch indices.
        query_words = query_words.repeat(batch_size, 1, 1)

        c_i = compute_region_context_vector(
            local_incept_feat, query_words, gamma1
        )  # Taken from attnGAN paper. shape: (batch, D, L)

        query_words = query_words.transpose(1, 2)  # shape: (batch, L, D)
        c_i = c_i.transpose(1, 2)  # shape: (batch, L, D)
        query_words = query_words.reshape(
            batch_size * numb_words, -1
        )  # shape: (batch * L, D)
        c_i = c_i.reshape(batch_size * numb_words, -1)  # shape: (batch * L, D)

        r_i = compute_relevance(
            c_i, query_words
        )  # cosine similarity, or R(c_i, e_i) from attnGAN paper. shape: (batch * L, 1)
        r_i = r_i.view(batch_size, numb_words)  # shape: (batch, L)
        r_i = torch.exp(r_i * gamma2)  # shape: (batch, L)
        r_i = r_i.sum(dim=1, keepdim=True)  # shape: (batch, 1)
        r_i = torch.log(
            r_i
        )  # This is image-text matching score b/w whole image and caption, shape: (batch, 1)
        match_scores.append(r_i)

    masks = torch.cat(masks, dim=0)  # type: ignore
    masks = torch.tensor(masks, dtype = torch.bool)  # type: ignore
    match_scores = torch.cat(match_scores, dim=1)  # type: ignore

    # This corresponds to P(D|Q) from attnGAN.
    match_scores = gamma3 * match_scores  # type: ignore
    match_scores.data.masked_fill_(  # type: ignore
        masks, -float("inf")
    )  # mask out the scores for mis-matched samples

    match_scores_t = match_scores.transpose(  # type: ignore
        0, 1
    )  # This corresponds to P(Q|D) from attnGAN.

    # This corresponds to L1_w from attnGAN.
    l1_w = nn.CrossEntropyLoss()(match_scores, match_labels)
    # This corresponds to L2_w from attnGAN.
    l2_w = nn.CrossEntropyLoss()(match_scores_t, match_labels)

    incept_feat_norm = torch.linalg.norm(global_incept_feat, dim=1)
    sent_emb_norm = torch.linalg.norm(sent_emb, dim=1)

    # shape: (batch, batch)
    global_match_score = global_incept_feat @ (sent_emb.T)

    global_match_score = (
        global_match_score / torch.outer(incept_feat_norm, sent_emb_norm)
    ).clamp(min=1e-8)
    global_match_score = gamma3 * global_match_score

    # mask out the scores for mis-matched samples
    global_match_score.data.masked_fill_(masks, -float("inf"))  # type: ignore

    global_match_t = global_match_score.T  # shape: (batch, batch)

    # This corresponds to L1_s from attnGAN.
    l1_s = nn.CrossEntropyLoss()(global_match_score, match_labels)
    # This corresponds to L2_s from attnGAN.
    l2_s = nn.CrossEntropyLoss()(global_match_t, match_labels)

    loss_damsm = lambda3 * (l1_w + l2_w + l1_s + l2_s)

    return loss_damsm


def compute_relevance(c_i: torch.Tensor, query_words: torch.Tensor) -> Any:
    """Computes the cosine similarity between the region context vector and the query words.

    Args:
        c_i: The region context vector. shape: (batch * L, D)
        query_words: The query words. shape: (batch * L, D)
    """
    prod = c_i * query_words  # shape: (batch * L, D)
    numr = torch.sum(prod, dim=1)  # shape: (batch * L, 1)
    norm_c = torch.linalg.norm(c_i, ord=2, dim=1)
    norm_q = torch.linalg.norm(query_words, ord=2, dim=1)
    denr = norm_c * norm_q
    r_i = (numr / denr).clamp(min=1e-8).squeeze()  # shape: (batch * L, 1)
    return r_i


def compute_region_context_vector(
    local_incept_feat: torch.Tensor, query_words: torch.Tensor, gamma1: float
) -> Any:
    """Compute the region context vector (c_i) from attnGAN paper.

    Args:
        local_incept_feat: The local inception features. [shape: (batch, D, 17, 17)]
        query_words: The embeddings for all the words in the captions. shape: (batch, D, L)
        gamma1: The gamma1 value from attnGAN paper.
    """
    batch, L = query_words.size(0), query_words.size(2)  # pylint: disable=invalid-name

    feat_height, feat_width = local_incept_feat.size(2), local_incept_feat.size(3)
    N = feat_height * feat_width  # pylint: disable=invalid-name

    # Reshape the local inception features to (batch, D, N)
    local_incept_feat = local_incept_feat.view(batch, -1, N)
    # shape: (batch, N, D)
    incept_feat_t = local_incept_feat.transpose(1, 2)

    sim_matrix = incept_feat_t @ query_words  # shape: (batch, N, L)
    sim_matrix = sim_matrix.view(batch * N, L)  # shape: (batch * N, L)

    sim_matrix = nn.Softmax(dim=1)(sim_matrix)  # shape: (batch * N, L)
    sim_matrix = sim_matrix.view(batch, N, L)  # shape: (batch, N, L)

    sim_matrix = torch.transpose(sim_matrix, 1, 2)  # shape: (batch, L, N)
    sim_matrix = sim_matrix.reshape(batch * L, N)  # shape: (batch * L, N)

    alpha_j = gamma1 * sim_matrix  # shape: (batch * L, N)
    alpha_j = nn.Softmax(dim=1)(alpha_j)  # shape: (batch * L, N)
    alpha_j = alpha_j.view(batch, L, N)  # shape: (batch, L, N)
    alpha_j_t = torch.transpose(alpha_j, 1, 2)  # shape: (batch, N, L)

    c_i = (
        local_incept_feat @ alpha_j_t
    )  # shape: (batch, D, L) [summing over N dimension in paper, so we multiply like this]
    return c_i


def discriminator_loss(
    logits: dict[str, dict[str, torch.Tensor]],
    labels: dict[str, dict[str, torch.Tensor]],
    lambda_4: float = 1.0,
) -> Any:
    """
    Calculate discriminator objective

    :param dict[str, dict[str, torch.Tensor]] logits:
        Dictionary with fake/real and word-level/uncond/cond logits

        Example:

        logits = {
            "fake": {
                "word_level": torch.Tensor (BxL)
                "uncond": torch.Tensor (Bx1)
                "cond": torch.Tensor (Bx1)
            },
            "real": {
                "word_level": torch.Tensor (BxL)
                "uncond": torch.Tensor (Bx1)
                "cond": torch.Tensor (Bx1)
            },
        }
    :param dict[str, dict[str, torch.Tensor]] labels:
        Dictionary with fake/real and word-level/image labels

        Example:

        labels = {
            "fake": {
                "word_level": torch.Tensor (BxL)
                "image": torch.Tensor (Bx1)
            },
            "real": {
                "word_level": torch.Tensor (BxL)
                "image": torch.Tensor (Bx1)
            },
        }
    :param float lambda_4: Hyperparameter for word loss in paper
    :return: Discriminator objective loss
    :rtype: Any
    """
    # define main loss functions for logit losses
    bce_logits = nn.BCEWithLogitsLoss()
    bce = nn.BCELoss()
    # calculate word-level loss
    word_loss = bce(logits["real"]["word_level"], labels["real"]["word_level"])
    word_loss += bce(logits["fake"]["word_level"], labels["fake"]["word_level"])
    # calculate unconditional adversarial loss
    uncond_loss = bce_logits(logits["real"]["uncond"], labels["real"]["image"])
    uncond_loss += bce_logits(logits["fake"]["uncond"], labels["fake"]["image"])
    # calculate conditional adversarial loss
    cond_loss = bce_logits(logits["real"]["cond"], labels["real"]["image"])
    cond_loss += bce_logits(logits["fake"]["cond"], labels["fake"]["image"])
    return -1 / 2 * (uncond_loss + cond_loss) + lambda_4 * word_loss


def kl_loss(mu_tensor: torch.Tensor, logvar: torch.Tensor) -> Any:
    """
    Calculate KL loss

    :param torch.Tensor mu_tensor: Mean of latent distribution
    :param torch.Tensor logvar: Log variance of latent distribution
    :return: KL loss [-0.5 * (1 + log(sigma) - mu^2 - sigma^2)]
    :rtype: Any
    """
    return torch.mean(-0.5 * (1 + 0.5 * logvar - mu_tensor.pow(2) - torch.exp(logvar)))
