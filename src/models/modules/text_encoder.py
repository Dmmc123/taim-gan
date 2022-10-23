"""LSTM-based textual encoder for tokenized input"""

from typing import Any

import torch
from torch import nn


class TextEncoder(nn.Module):
    """Simple text encoder based on RNN"""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        """
        Initialize embeddings lookup for tokens and main LSTM

        :param vocab_size:
            Size of created vocabulary for textual input. L from paper
        :param emb_dim: Length of embeddings for each word.
        :param hidden_dim:
            Length of hidden state of a LSTM cell. 2 x hidden_dim = C (from LWGAN paper)
        """
        super().__init__()
        self.embs = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> Any:
        """
        Propagate the text token input through the LSTM and return
        two types of embeddings: word-level and sentence-level

        :param torch.Tensor tokens: Input text tokens from vocab
        :return: Word-level embeddings (BxCxL) and sentence-level embeddings (BxC)
        :rtype: Any
        """
        embs = self.embs(tokens)
        output, (hidden_states, _) = self.lstm(embs)
        word_embs = torch.transpose(output, 1, 2)
        sent_embs = torch.cat((hidden_states[-1, :, :], hidden_states[0, :, :]), dim=1)
        return word_embs, sent_embs
