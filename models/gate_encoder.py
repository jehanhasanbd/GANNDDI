# models/gate_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention, CrossAttention


class GATEEncoderBlock(nn.Module):
    """GATE Encoder block with gated attention"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout
        )

        # Cross attention
        self.cross_attention = CrossAttention(
            hidden_dim, num_heads, dropout
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        x: node features [batch_size, num_nodes, hidden_dim]
        y: cross-attention context [batch_size, num_patterns, hidden_dim]
        """

        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention if context provided
        if y is not None:
            cross_output = self.cross_attention(x, y, mask)
            x = self.norm2(x + self.dropout(cross_output))

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class GATEEncoder(nn.Module):
    """GATE Encoder for drug interactions"""

    def __init__(
            self,
            hidden_dim: int = 256,
            num_heads: int = 8,
            num_layers: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create encoder blocks
        self.blocks = nn.ModuleList([
            GATEEncoderBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Learnable query prototypes for cross-attention
        self.query_prototypes = nn.Parameter(
            torch.randn(1, 16, hidden_dim) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            node_features: torch.Tensor,
            drug_features: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        node_features: [batch_size, num_nodes, hidden_dim]
        drug_features: [batch_size, feat_dim] (optional graph-level features)
        """

        batch_size = node_features.size(0)
        num_nodes = node_features.size(1)

        # Expand query prototypes for batch
        queries = self.query_prototypes.expand(batch_size, -1, -1)  # [batch_size, num_patterns, hidden_dim]

        # Process through encoder blocks
        x = node_features

        for block in self.blocks:
            x = block(x, queries, mask)

        # Pool node features
        # Use attention-based pooling or mean pooling
        pooled = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]

        # Combine with drug features if provided
        if drug_features is not None:
            if drug_features.dim() == 2:
                drug_features = drug_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            pooled = pooled + drug_features

        # Return both node-level and graph-level representations
        return x, pooled.squeeze(1)  # [batch_size, num_nodes, hidden_dim], [batch_size, hidden_dim]