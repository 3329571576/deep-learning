"""Bi-LSTM 情感分类模型，支持多种池化方式。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPooling(nn.Module):
    """对 Bi-LSTM 输出做注意力加权池化，比仅取最后隐状态更适合长文本。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # output: [B, T, 2H], mask: [B, T] (True 表示有效 token)
        scores = self.proj(output).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), output).squeeze(1)


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        embed_dropout: float = 0.0,
        pooling: str = "last",
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.pooling = pooling

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
        )
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionPooling(hidden_dim) if pooling == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def _make_mask(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = input_ids.size(1)
        range_idx = torch.arange(max_len, device=input_ids.device).unsqueeze(0)
        return range_idx < lengths.unsqueeze(1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embed_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, _) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.pooling == "attention":
            mask = self._make_mask(input_ids, lengths)
            context = self.attention(output, mask)
        else:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            context = torch.cat([forward_hidden, backward_hidden], dim=1)

        context = self.dropout(context)
        return self.fc(context)
