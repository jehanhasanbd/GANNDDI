# models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with optional mask"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.1,
            bias: bool = True
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = query.size(0)

        # Linear projections
        Q = self.query(query).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = self.key(key).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = self.value(value).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )

        output = self.out(context)

        if return_attention:
            return output, attention_weights

        return output, None


class CrossAttention(nn.Module):
    """Cross-attention module between two sequences"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Cross-attention: attend y to x
        attn_output, _ = self.attention(x, y, y, mask)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x