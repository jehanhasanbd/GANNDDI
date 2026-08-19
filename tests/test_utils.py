# tests/test_utils.py

import torch
import numpy as np
import pytest

from utils.data_utils import DataUtils
from utils.chem_utils import ChemistryUtils
from utils.metrics import MetricsCalculator

class TestUtils:
    """Test cases for utility functions"""

    def setup_method(self):
        self.y_true = torch.tensor([0, 1, 0, 1, 1])
        self.y_pred = torch.tensor([0, 1, 1, 0, 1])
        self.y_score = torch.tensor([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.1, 0.9]
        ])

    def test_metrics(self):
        """Test metrics calculation"""
        metrics = MetricsCalculator.compute_all_metrics(
            self.y_true,
            self.y_pred,
            self.y_score
        )

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auroc' in metrics
        assert 'auprc' in metrics

        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auroc'] <= 1
        assert 0 <= metrics['auprc'] <= 1

    def test_chemistry_utils(self):
        """Test chemistry utilities"""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

        assert ChemistryUtils.validate_smiles(smiles) is True
        assert ChemistryUtils.validate_smiles("invalid_smiles") is False

        mol_weight = ChemistryUtils.get_molecular_weight(smiles)
        assert mol_weight is not None
        assert mol_weight > 0

        atom_count = ChemistryUtils.get_atom_count(smiles)
        assert atom_count is not None
        assert atom_count > 0

        fingerprint = ChemistryUtils.get_morgan_fingerprint(smiles)
        assert fingerprint is not None

    def test_data_utils(self):
        """Test data utilities"""
        # Test tensor conversion
        data = np.array([1, 2, 3])
        tensor = DataUtils.to_tensor(data)
        assert isinstance(tensor, torch.Tensor)

        # Test numpy conversion
        np_data = DataUtils.to_numpy(tensor)
        assert isinstance(np_data, np.ndarray)

        # Test JSON serialization
        test_data = {'a': 1, 'b': 2}
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            DataUtils.save_json(test_data, f.name)
            loaded = DataUtils.load_json(f.name)
            assert loaded == test_data
            os.unlink(f.name)