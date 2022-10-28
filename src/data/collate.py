"""Custom collate function for the data loader."""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_collate(batch: list[Any]) -> Any:
    """
    Custom collate function to be used in the data loader.
    :param batch: list, with length equal to number of batches.
    :return: processed batch of data [add padding to text, stack tensors in batch]
    """
    img, correct_capt, curr_class, wrong_capt, wrong_class, word_labels = zip(*batch)
    batched_img = torch.stack(img, dim=0).to(
        device
    )  # shape: (batch_size, 3, height, width)
    correct_capt_len = torch.tensor(
        [len(capt) for capt in correct_capt], dtype=torch.int64
    ).unsqueeze(
        1
    )  # shape: (batch_size, 1)
    batched_correct_capt = pad_sequence(
        correct_capt, batch_first=True, padding_value=0
    ).to(
        device
    )  # shape: (batch_size, max_seq_len)
    batched_curr_class = torch.stack(curr_class, dim=0).to(
        device
    )  # shape: (batch_size, 1)
    wrong_capt_len = torch.tensor(
        [len(capt) for capt in wrong_capt], dtype=torch.int64
    ).unsqueeze(
        1
    )  # shape: (batch_size, 1)
    batched_wrong_capt = pad_sequence(wrong_capt, batch_first=True, padding_value=0).to(
        device
    )  # shape: (batch_size, max_seq_len)
    batched_wrong_class = torch.stack(wrong_class, dim=0).to(
        device
    )  # shape: (batch_size, 1)
    batched_word_labels = pad_sequence(
        word_labels, batch_first=True, padding_value=0
    ).to(
        device
    )  # shape: (batch_size, max_seq_len)
    return (
        batched_img,
        batched_correct_capt,
        correct_capt_len,
        batched_curr_class,
        batched_wrong_capt,
        wrong_capt_len,
        batched_wrong_class,
        batched_word_labels,
    )
