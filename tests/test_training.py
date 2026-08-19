# tests/test_training.py

"""
Test training module
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.trainer import Trainer, Evaluator, DDILoss
from training.loss_functions import FocalLoss
from models.ddi_predictor import DDIPredictor
from data.dataset import create_dataloaders, MolecularGraph


class TestTraining:
    """Test training module"""

    def setup_method(self):
        self.hidden_dim = 64
        self.num_classes = 5
        self.batch_size = 4
        self.num_nodes = 10

    def test_trainer_initialization(self, config, device):
        """Test trainer initialization"""
        model = DDIPredictor(
            node_feat_dim=74,
            edge_feat_dim=10,
            hidden_dim=64,
            num_patterns=8,
            num_heads=4,
            num_classes=10
        ).to(device)

        trainer = Trainer(
            model=model,
            config=config.model.__dict__ if hasattr(config.model, '__dict__') else {},
            device=str(device)
        )

        assert trainer is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None

    def test_loss_functions(self):
        """Test loss functions"""
        batch_size = 16
        num_classes = 10

        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))

        # Test focal loss
        focal_loss = FocalLoss(gamma=2.0)
        loss = focal_loss(logits, labels)
        assert loss.item() >= 0

        # Test DDI loss
        ddi_loss = DDILoss(num_classes=num_classes)
        loss = ddi_loss(logits, labels)
        assert loss.item() >= 0

    def test_evaluator(self, config, device):
        """Test evaluator"""
        evaluator = Evaluator(config.model.__dict__ if hasattr(config.model, '__dict__') else {})

        # Create dummy model
        model = DDIPredictor(
            node_feat_dim=74,
            edge_feat_dim=10,
            hidden_dim=64,
            num_patterns=8,
            num_heads=4,
            num_classes=10
        ).to(device)

        # Create dummy dataloader
        class DummyDataset:
            def __init__(self):
                self.batch_size = 4

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    'node_features1': torch.randn(10, 74),
                    'edge_index1': torch.randint(0, 10, (2, 5)),
                    'edge_features1': torch.randn(5, 10),
                    'node_features2': torch.randn(10, 74),
                    'edge_index2': torch.randint(0, 10, (2, 5)),
                    'edge_features2': torch.randn(5, 10),
                    'labels': torch.randint(0, 10, (1,))
                }

        # Note: This is a simplified test - actual evaluation would need proper data
        assert evaluator is not None