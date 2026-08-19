# tests/test_data.py

import pytest
import torch
import numpy as np
from rdkit import Chem

from data.drugbank_loader import (
    MolecularGraph,
    DrugBankLoader,
    Preprocessor
)
from data.dataset import DDIDataset
from config.model_config import get_default_config


class TestData:
    """Test cases for data module"""

    def setup_method(self):
        self.smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        self.drug_id = "DB00001"
        self.config = get_default_config()

    def test_molecular_graph_creation(self):
        """Test molecular graph creation from SMILES"""
        graph = MolecularGraph.from_smiles(self.smiles, self.drug_id)

        assert graph is not None
        assert graph.smiles == self.smiles
        assert graph.drug_id == self.drug_id
        assert graph.num_atoms > 0
        assert graph.num_bonds > 0
        assert graph.node_features.shape[0] == 150  # max_atoms
        assert graph.edge_index.shape[0] == 2

    def test_graph_collate_fn(self):
        """Test graph collate function"""
        graph1 = MolecularGraph.from_smiles(self.smiles, "DB00001")
        graph2 = MolecularGraph.from_smiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "DB00002")

        batch = [
            {'graph': graph1, 'label': 0},
            {'graph': graph2, 'label': 1}
        ]

        from data.dataset import graph_collate_fn
        batched = graph_collate_fn(batch)

        assert 'node_features' in batched
        assert 'edge_index' in batched
        assert 'labels' in batched
        assert batched['labels'].shape == (2,)
        assert batched['node_features'].shape[0] == 2

    def test_ddi_dataset(self):
        """Test DDI dataset"""
        data = [
            {
                'drug1_id': 'DB00001',
                'drug2_id': 'DB00002',
                'label': 0
            }
        ]

        drug_smiles = {
            'DB00001': self.smiles,
            'DB00002': "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        }

        dataset = DDIDataset(data, drug_smiles, max_atoms=150, precompute_graphs=True)

        assert len(dataset) == 1

        sample = dataset[0]
        assert 'graph1' in sample
        assert 'graph2' in sample
        assert 'label' in sample
        assert sample['label'] == 0

    def test_preprocessor(self):
        """Test preprocessor"""
        preprocessor = Preprocessor(self.config)

        # Test feature extraction
        from data.molecular_graph import MolecularGraph
        graph = MolecularGraph.from_smiles(self.smiles, self.drug_id)

        assert graph.node_features is not None
        assert graph.edge_index is not None
        assert graph.graph_features is not None