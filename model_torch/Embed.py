import torch
import torch.nn as nn
import math


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Vectorized construction of the positional encoding matrix
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()          # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )                                                                       # (d_model/2,)

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale embeddings before adding positional encoding
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        # pe is already on the correct device via the registered buffer
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
