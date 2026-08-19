# tests/test_models.py

import torch
import pytest
import numpy as np

from models.ddi_predictor import DDIPredictor
from models.sie_encoder import SIEEncoder
from models.gaan import GatedAttentionNetwork
from models.gate_encoder import GATEEncoder
from modules.multi_head_attention import (
    MultiHeadAttention,
    GatedAttention,
    GlobalMeanPool
)



class TestModels:
    """Test cases for models"""

    def setup_method(self):
        self.batch_size = 4
        self.num_nodes = 10
        self.hidden_dim = 64
        self.num_heads = 4
        self.num_patterns = 8
        self.num_classes = 10

        # Create dummy inputs
        self.node_features = torch.randn(self.batch_size, self.num_nodes, 74)
        self.edge_index = torch.randint(0, self.num_nodes, (self.batch_size, 2, 15))
        self.edge_features = torch.randn(self.batch_size, 15, 10)
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_gaan(self):
        """Test Gated Attention Network"""
        model = GatedAttentionNetwork(
            node_feat_dim=74,
            edge_feat_dim=10,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=3,
            num_heads=self.num_heads
        )

        output = model(
            self.node_features,
            self.edge_index,
            self.edge_features
        )

        assert output.shape == (self.batch_size, self.hidden_dim)

    def test_gate_encoder(self):
        """Test GATE Encoder"""
        model = GATEEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=2
        )

        # Reshape node features
        node_feats = self.node_features.view(
            self.batch_size, self.num_nodes, 74
        )

        # Project to hidden_dim
        proj = torch.nn.Linear(74, self.hidden_dim)
        node_feats = proj(node_feats)

        node_output, graph_output = model(node_feats)

        assert node_output.shape == (self.batch_size, self.num_nodes, self.hidden_dim)
        assert graph_output.shape == (self.batch_size, self.hidden_dim)

    def test_sie_encoder(self):
        """Test SIE Encoder"""
        model = SIEEncoder(
            hidden_dim=self.hidden_dim,
            num_patterns=self.num_patterns
        )

        # Project node features
        proj = torch.nn.Linear(74, self.hidden_dim)
        node_feats = proj(self.node_features)

        patterns = model(node_feats)
        assert patterns.shape == (self.batch_size, self.num_patterns, self.hidden_dim)

        # Test similarity computation
        patterns2 = torch.randn(self.batch_size, self.num_patterns, self.hidden_dim)
        similarity = model.compute_similarity(patterns, patterns2)
        assert similarity.shape == (self.batch_size, self.num_patterns, self.num_patterns)
        assert torch.all(similarity >= -1) and torch.all(similarity <= 1)

    def test_ddi_predictor(self):
        """Test DDI Predictor"""
        model = DDIPredictor(
            node_feat_dim=74,
            edge_feat_dim=10,
            hidden_dim=self.hidden_dim,
            num_patterns=self.num_patterns,
            num_heads=self.num_heads,
            num_classes=self.num_classes
        )

        logits = model(
            self.node_features,
            self.edge_index,
            self.edge_features,
            self.node_features,
            self.edge_index,
            self.edge_features
        )

        assert logits.shape == (self.batch_size, self.num_classes)

        # Test prediction
        probs = model.predict(
            self.node_features,
            self.edge_index,
            self.edge_features,
            self.node_features,
            self.edge_index,
            self.edge_features
        )

        assert probs.shape == (self.batch_size, self.num_classes)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)