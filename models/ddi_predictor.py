# models/ddi_predictor.py

"""
DDI Prediction module combining GATE and SIE encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.gaan import GatedAttentionNetwork
from models.gate_encoder import GATEEncoder
from models.sie_encoder import SIEEncoder

"""
DDI Prediction module combining GATE and SIE encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DDIPredictor(nn.Module):
    """DDI Prediction module combining GATE and SIE encoders"""

    def __init__(
            self,
            node_feat_dim: Optional[int] = None,  # Make optional
            edge_feat_dim: int = 10,
            hidden_dim: int = 256,
            output_dim: int = 128,
            num_patterns: int = 16,
            num_heads: int = 8,
            gaan_layers: int = 3,
            gate_layers: int = 2,
            dropout: float = 0.1,
            num_classes: int = 65  # Number of DDI event types
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_patterns = num_patterns
        self.node_feat_dim = node_feat_dim

        # GaAN for node embeddings (will handle dynamic dimensions)
        self.gaan = GatedAttentionNetwork(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=gaan_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # GATE Encoder
        self.gate_encoder = GATEEncoder(
            hidden_dim=output_dim,
            num_heads=num_heads,
            num_layers=gate_layers,
            dropout=dropout
        )

        # SIE Encoder
        self.sie_encoder = SIEEncoder(
            hidden_dim=output_dim,
            num_patterns=num_patterns,
            dropout=dropout
        )

        # Interaction classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 2 + num_patterns * num_patterns, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            node_features1: torch.Tensor,
            edge_index1: torch.Tensor,
            edge_features1: Optional[torch.Tensor],
            node_features2: torch.Tensor,
            edge_index2: torch.Tensor,
            edge_features2: Optional[torch.Tensor],
            batch1: Optional[torch.Tensor] = None,
            batch2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for DDI prediction
        """

        batch_size = node_features1.size(0)
        node_feat_dim = node_features1.size(2)

        # Get graph-level embeddings from GaAN
        graph1_embeds = self.gaan(node_features1, edge_index1, edge_features1, batch1)
        graph2_embeds = self.gaan(node_features2, edge_index2, edge_features2, batch2)

        # For GATE and SIE, project node features to output_dim
        # Create projection layer if needed
        if not hasattr(self, 'node_proj'):
            self.node_proj = nn.Linear(node_feat_dim, self.output_dim).to(node_features1.device)

        # Project node features
        node_feats1 = self.node_proj(node_features1)  # [batch_size, max_nodes, output_dim]
        node_feats2 = self.node_proj(node_features2)  # [batch_size, max_nodes, output_dim]

        # GATE encoder
        _, gate_output1 = self.gate_encoder(node_feats1)
        _, gate_output2 = self.gate_encoder(node_feats2)

        # SIE encoder
        patterns1 = self.sie_encoder(node_feats1)
        patterns2 = self.sie_encoder(node_feats2)

        # Compute similarity matrix
        similarity = self.sie_encoder.compute_similarity(patterns1, patterns2)

        # Flatten similarity
        sim_flat = similarity.view(batch_size, -1)  # [batch_size, num_patterns * num_patterns]

        # Combine features
        combined = torch.cat([
            gate_output1,  # [batch_size, output_dim]
            gate_output2,  # [batch_size, output_dim]
            sim_flat  # [batch_size, num_patterns * num_patterns]
        ], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Predict interaction probabilities"""
        logits = self.forward(*args, **kwargs)
        return F.softmax(logits, dim=-1)