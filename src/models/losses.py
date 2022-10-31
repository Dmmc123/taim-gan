"""Module containing the loss functions for the GANs."""
from typing import Any

import torch
from torch import nn

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


def generator_loss(
    discriminator: Any,
    inception_encoder: Any,
    fake_imgs: torch.Tensor,
    real_labels: torch.Tensor,
    words_emb: torch.Tensor,
    sent_emb: torch.Tensor,
    match_labels: torch.Tensor,
    cap_lens: torch.Tensor,
    class_ids: torch.Tensor,
    vgg_encoder: Any,
    real_imgs: torch.Tensor,
    device: Any,
    const_dict: dict[Any, Any],
) -> Any:
    """Calculate the loss for the generator.

    Args:
        discriminator: The discriminator model.

        inception_encoder: The inception encoder model.

        fake_imgs: The fake images generated by the generator. [shape: (batch_size, 3, 256, 256)]

        real_labels: Label for "real" image as predicted by discriminator,
        this is a tensor of ones. [shape: (batch_size, 1)].

        words_emb: The embeddings for all the words in the captions.
        shape: (batch_size, embedding_size, max_caption_length)

        sent_emb: The embeddings for the sentences.
        shape: (batch_size, embedding_size)

        match_labels: Tensor of shape: (batch_size, 1).
        This is of the form torch.tensor([0, 1, 2, ..., batch-1])

        cap_lens: The length of the 'actual' captions in the batch [without padding]
        shape: (batch_size, 1)

        class_ids: The class ids for the instance. shape: (batch_size, 1)

        vgg_encoder: The VGG encoder model.

        real_imgs: The Real Images from the dataset. shape: (batch_size, 3, 256, 256)

        device: The device to run the model on.

        const_dict: The dictionary containing the constants.
    """
    total_error_g = 0
    lambda3 = const_dict["lambda3"]

    cond_logits = discriminator.COND_DNET(fake_imgs, sent_emb)
    cond_err_g = binary_cross_entropy(cond_logits, real_labels)

    uncond_logits = discriminator.UNCOND_DNET(fake_imgs)
    uncond_err_g = binary_cross_entropy(uncond_logits, real_labels)

    loss_g = (
        cond_err_g + uncond_err_g
    )  # add up the conditional and unconditional losses
    total_error_g += loss_g

    local_incept_feat, global_incept_feat = inception_encoder(fake_imgs)
    l1_w, l2_w, l1_s, l2_s = damsm_loss(
        local_incept_feat,
        global_incept_feat,
        words_emb,
        sent_emb,
        match_labels,
        cap_lens,
        class_ids,
        device,
        const_dict,
    )  # DAMSM Loss from attnGAN.

    loss_damsm = lambda3 * (l1_w + l2_w + l1_s + l2_s)

    total_error_g += loss_damsm

    vgg_real = vgg_encoder(real_imgs)  # shape: (batch, 128, 128, 128)
    vgg_fake = vgg_encoder(fake_imgs)  # shape: (batch, 128, 128, 128)

    loss_per = mean_squared_error(vgg_real, vgg_fake)

    total_error_g += loss_per / 2.0
    return total_error_g


def damsm_loss(
    local_incept_feat: torch.Tensor,
    global_incept_feat: torch.Tensor,
    words_emb: torch.Tensor,
    sent_emb: torch.Tensor,
    match_labels: torch.Tensor,
    cap_lens: torch.Tensor,
    class_ids: torch.Tensor,
    device: Any,
    const_dict: dict[Any, Any],
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

        device: The device to run the model on.

        const_dict: The dictionary containing the constants.
    """
    batch_size = match_labels.size(0)
    masks = (
        []
    )  # Mask mis-match samples, that come from the same class as the real sample ###

    match_scores = []
    cap_len_list = cap_lens.data.tolist()
    gamma1 = const_dict["gamma1"]
    gamma2 = const_dict["gamma2"]
    gamma3 = const_dict["gamma3"]

    for i in range(batch_size):
        mask = (class_ids == class_ids[i]).type(torch.int)
        mask[
            i
        ] = 0  # This ensures that "correct class" index is not included in the mask.
        masks.append(mask.reshape(1, -1))  # shape: (1, batch)

        numb_words = cap_len_list[i]
        query_words = (
            words_emb[i, :, :numb_words].unsqueeze(0).contiguous()
        )  # shape: (1, D, L), this picks the caption at ith batch index.
        query_words = query_words.repeat(
            batch_size, 1, 1
        )  # shape: (batch, D, L), this expands the same caption for all batch indices.

        c_i = compute_region_context_vector(
            local_incept_feat, query_words, gamma1
        )  # Taken from attnGAN paper. shape: (batch, D, L)

        query_words = query_words.transpose(1, 2).contiguous()  # shape: (batch, L, D)
        c_i = c_i.transpose(1, 2).contiguous()  # shape: (batch, L, D)
        query_words = query_words.view(
            batch_size * numb_words, -1
        )  # shape: (batch * L, D)
        c_i = c_i.view(batch_size * numb_words, -1)  # shape: (batch * L, D)

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
    masks = torch.BoolTensor(masks).to(device)  # type: ignore
    match_scores = torch.cat(match_scores, dim=1)  # type: ignore

    match_scores = gamma3 * match_scores  # This corresponds to P(D|Q) from attnGAN.
    match_scores.data.masked_fill_(  # type: ignore
        masks, -float("inf")
    )  # mask out the scores for mis-matched samples

    match_scores_t = match_scores.transpose(  # type: ignore
        0, 1
    )  # This corresponds to P(Q|D) from attnGAN.

    l1_w = cross_entropy(
        match_scores, match_labels  # type: ignore
    )  # This corresponds to L1_w from attnGAN.
    l2_w = cross_entropy(
        match_scores_t, match_labels
    )  # This corresponds to L2_w from attnGAN.

    global_incept_feat = global_incept_feat.unsqueeze(0)  # shape: (1, batch, D)
    sent_emb = sent_emb.unsqueeze(0)  # shape: (1, batch, D)

    incept_feat_norm = torch.norm(
        global_incept_feat, 2, dim=2, keepdim=True
    )  # shape: (1, batch, 1)
    sent_emb_norm = torch.norm(sent_emb, 2, dim=2, keepdim=True)  # shape: (1, batch, 1)

    global_match_score = global_incept_feat @ (
        sent_emb.transpose(1, 2)
    )  # shape: (1, batch, batch)
    global_match_norm = incept_feat_norm @ (
        sent_emb_norm.transpose(1, 2)
    )  # shape: (1, batch, batch)

    global_match_score = (global_match_score / global_match_norm).clamp(min=1e-8)
    global_match_score = gamma3 * global_match_score

    global_match_score = global_match_score.squeeze()  # shape: (batch, batch)
    global_match_score.data.masked_fill_(  # type: ignore
        masks, -float("inf")
    )  # mask out the scores for mis-matched samples

    global_match_t = global_match_score.transpose(0, 1)  # shape: (batch, batch)

    l1_s = cross_entropy(
        global_match_score, match_labels
    )  # This corresponds to L1_s from attnGAN.
    l2_s = cross_entropy(
        global_match_t, match_labels
    )  # This corresponds to L2_s from attnGAN.

    return l1_w, l2_w, l1_s, l2_s


def compute_relevance(c_i: torch.Tensor, query_words: torch.Tensor) -> Any:
    """Computes the cosine similarity between the region context vector and the query words.

    Args:
        c_i: The region context vector. shape: (batch * L, D)
        query_words: The query words. shape: (batch * L, D)
    """
    prod = c_i * query_words  # shape: (batch * L, D)
    numr = torch.sum(prod, dim=1)  # shape: (batch * L, 1)
    norm_c = torch.norm(c_i, 2, dim=1)
    norm_q = torch.norm(query_words, 2, dim=1)
    denr = norm_c * norm_q
    r_i = (numr / denr).clamp(min=1e-8).squeeze()  # shape: (batch * L, 1)
    return r_i


def compute_region_context_vector(
    local_incept_feat: torch.Tensor, query_words: torch.Tensor, gamma1: int
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
    incept_feat_t = local_incept_feat.transpose(
        1, 2
    ).contiguous()  # shape: (batch, N, D)

    sim_matrix = incept_feat_t @ query_words  # shape: (batch, N, L)
    sim_matrix = sim_matrix.view(batch * N, L)  # shape: (batch * N, L)

    sim_matrix = nn.Softmax(dim=1)(sim_matrix)  # shape: (batch * N, L)
    sim_matrix = sim_matrix.view(batch, N, L)  # shape: (batch, N, L)

    sim_matrix = torch.transpose(sim_matrix, 1, 2)  # shape: (batch, L, N)
    sim_matrix = sim_matrix.view(batch * L, N)  # shape: (batch * L, N)

    alpha_j = gamma1 * sim_matrix  # shape: (batch * L, N)
    alpha_j = nn.Softmax(dim=1)(alpha_j)  # shape: (batch * L, N)
    alpha_j = alpha_j.view(batch, L, N)  # shape: (batch, L, N)
    alpha_j_t = torch.transpose(alpha_j, 1, 2)  # shape: (batch, N, L)

    c_i = (
        local_incept_feat @ alpha_j_t
    )  # shape: (batch, D, L) [summing over N dimension in paper, so we multiply like this]
    return c_i


def mean_squared_error(input_tensor: torch.Tensor, target: torch.Tensor) -> Any:
    """Computes the mean squared error between two tensors.

    Args:
        input_tensor: The input tensor.
        target: The target tensor.
    """
    return nn.MSELoss()(input_tensor, target)


def cross_entropy(input_tensor: torch.Tensor, target: torch.Tensor) -> Any:
    """Computes the cross entropy loss between the input and the target.

    Args:
        input_tensor: The input tensor.
        target: The target tensor.
    """
    return nn.CrossEntropyLoss()(input_tensor, target)


def binary_cross_entropy(input_tensor: torch.Tensor, target: torch.Tensor) -> Any:
    """Calculate the binary cross entropy loss.

    Args:
        input_tensor: The input tensor.
        target: The target tensor.

    Returns:
        The binary cross entropy loss.
    """
    return nn.BCELoss()(input_tensor, target)

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
