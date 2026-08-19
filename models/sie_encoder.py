# models/sie_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class SIEEncoder(nn.Module):
    """Similarity Identifier Encoder with self-attention and cosine similarity"""

    def __init__(
            self,
            hidden_dim: int = 256,
            num_patterns: int = 16,
            dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns

        # Learnable query patterns
        self.query_patterns = nn.Parameter(
            torch.randn(1, num_patterns, hidden_dim) * 0.02
        )

        # Projection layers
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(
            self,
            node_features: torch.Tensor,
            return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        node_features: [batch_size, num_nodes, hidden_dim]
        return_attention: whether to return attention weights

        Returns:
            patterns: [batch_size, num_patterns, hidden_dim]
            attention: [batch_size, num_patterns, num_nodes] (optional)
        """

        batch_size = node_features.size(0)
        num_nodes = node_features.size(1)

        # Expand query patterns for batch
        Q = self.query_patterns.expand(batch_size, -1, -1)  # [batch_size, num_patterns, hidden_dim]
        Q = self.query_proj(Q)  # [batch_size, num_patterns, hidden_dim]

        # Keys and values from node features
        K = self.key_proj(node_features)  # [batch_size, num_nodes, hidden_dim]
        V = self.value_proj(node_features)  # [batch_size, num_nodes, hidden_dim]

        # Attention scores
        # [batch_size, num_patterns, num_nodes]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        # [batch_size, num_patterns, hidden_dim]
        patterns = torch.matmul(attention_weights, V)
        patterns = self.output_proj(patterns)
        patterns = self.norm(patterns + self.dropout(patterns))

        if return_attention:
            return patterns, attention_weights

        return patterns

    def compute_similarity(
            self,
            patterns1: torch.Tensor,
            patterns2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of patterns
        patterns1: [batch_size, num_patterns, hidden_dim]
        patterns2: [batch_size, num_patterns, hidden_dim]

        Returns:
            similarity: [batch_size, num_patterns, num_patterns]
        """

        # Normalize patterns
        patterns1_norm = F.normalize(patterns1, p=2, dim=-1)
        patterns2_norm = F.normalize(patterns2, p=2, dim=-1)

        # Cosine similarity matrix
        similarity = torch.matmul(patterns1_norm, patterns2_norm.transpose(-2, -1))

        return similarity