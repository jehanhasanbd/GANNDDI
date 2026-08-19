# modules/multi_head_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class AttentionHead(nn.Module):
    """Single attention head"""

    def __init__(
            self,
            hidden_dim: int,
            head_dim: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.head_dim = head_dim

        self.query = nn.Linear(hidden_dim, head_dim, bias=False)
        self.key = nn.Linear(hidden_dim, head_dim, bias=False)
        self.value = nn.Linear(hidden_dim, head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: [batch_size, query_len, hidden_dim]
        key: [batch_size, key_len, hidden_dim]
        value: [batch_size, value_len, hidden_dim]
        """

        Q = self.query(query)  # [batch_size, query_len, head_dim]
        K = self.key(key)  # [batch_size, key_len, head_dim]
        V = self.value(value)  # [batch_size, value_len, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [batch_size, query_len, head_dim]

        return out, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

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
        query_len = query.size(1)
        key_len = key.size(1)
        value_len = value.size(1)

        # Linear projections
        Q = self.query(query).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_heads, query_len, head_dim]

        K = self.key(key).view(
            batch_size, key_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_heads, key_len, head_dim]

        V = self.value(value).view(
            batch_size, value_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_heads, value_len, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask shape: [batch_size, query_len, key_len]
            # Expand to [batch_size, num_heads, query_len, key_len]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, query_len, key_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.hidden_dim
        )

        output = self.out(context)

        if return_attention:
            return output, attn_weights

        return output, None


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x, mask)
        return self.norm(x + self.dropout(attn_output))


class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if value is None:
            value = key

        attn_output, _ = self.attention(query, key, value, mask)
        return self.norm(query + self.dropout(attn_output))