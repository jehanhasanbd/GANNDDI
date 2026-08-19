# tests/conftest.py

"""
Pytest configuration and fixtures
"""
import pytest
import torch
import numpy as np
from rdkit import Chem
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.model_config import get_default_config
from data.drugbank_loader import MolecularGraph, DrugBankLoader


@pytest.fixture
def device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def config():
    """Get default configuration"""
    return get_default_config()


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings"""
    return {
        'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'paracetamol': 'CC(=O)NC1=CC=C(C=C1)O',
        'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    }


@pytest.fixture
def sample_molecular_graph(sample_smiles):
    """Create sample molecular graph"""
    smiles = sample_smiles['aspirin']
    return MolecularGraph.from_smiles(smiles, 'DB00001')


@pytest.fixture
def sample_batch(sample_smiles):
    """Create sample batch data"""
    batch_size = 4
    num_nodes = 20
    hidden_dim = 64

    return {
        'node_features1': torch.randn(batch_size, num_nodes, 74),
        'edge_index1': torch.randint(0, num_nodes, (batch_size, 2, 15)),
        'edge_features1': torch.randn(batch_size, 15, 10),
        'node_features2': torch.randn(batch_size, num_nodes, 74),
        'edge_index2': torch.randint(0, num_nodes, (batch_size, 2, 15)),
        'edge_features2': torch.randn(batch_size, 15, 10),
        'labels': torch.randint(0, 10, (batch_size,))
    }


@pytest.fixture
def mock_drugbank_data():
    """Mock DrugBank data for testing"""
    return {
        'train_data': [
            {'drug1_id': 'DB00001', 'drug2_id': 'DB00002', 'label': 0},
            {'drug1_id': 'DB00003', 'drug2_id': 'DB00004', 'label': 1},
        ],
        'val_data': [
            {'drug1_id': 'DB00005', 'drug2_id': 'DB00006', 'label': 0},
        ],
        'test_data': [
            {'drug1_id': 'DB00007', 'drug2_id': 'DB00008', 'label': 1},
        ],
        'drug_smiles': {
            'DB00001': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'DB00002': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'DB00003': 'CC(=O)NC1=CC=C(C=C1)O',
            'DB00004': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'DB00005': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'DB00006': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'DB00007': 'CC(=O)NC1=CC=C(C=C1)O',
            'DB00008': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        },
        'drug_info': {
            'DB00001': {'name': 'Aspirin', 'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'},
            'DB00002': {'name': 'Ibuprofen', 'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'},
        },
        'label_map': {'interaction_type_1': 0, 'interaction_type_2': 1}
    }