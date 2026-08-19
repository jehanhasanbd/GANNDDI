"""
Dataset module for GANNDDI - Memory Efficient Version
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import gc

from data.molecular_graph import MolecularGraph, graph_collate_fn


class DDIDataset(Dataset):
    """
    Memory-efficient Dataset for DDI prediction with on-the-fly graph creation.
    Precomputes graphs only for small datasets to save memory.
    """

    def __init__(
            self,
            data: List[Dict],
            drug_smiles: Dict[str, str],
            max_atoms: int = 100,
            precompute_graphs: bool = False  # Default False to save memory
    ):
        """
        Initialize dataset.

        Args:
            data: List of dictionaries containing drug pairs and labels
            drug_smiles: Dictionary mapping drug_id to SMILES string
            max_atoms: Maximum number of atoms per molecule
            precompute_graphs: Whether to precompute all graphs (only for small datasets)
        """
        self.data = data
        self.drug_smiles = drug_smiles
        self.max_atoms = max_atoms
        self.graph_cache = {}
        self.precompute_graphs = precompute_graphs

        # Only precompute if dataset is small (less than 1000 samples)
        if precompute_graphs and len(data) < 1000:
            self._precompute_graphs()

        print(f"Dataset initialized with {len(data)} samples (precompute={precompute_graphs})")

    def _precompute_graphs(self):
        """Precompute molecular graphs for all drugs (only for small datasets)"""
        unique_drugs = set()
        for entry in self.data:
            unique_drugs.add(entry['drug1_id'])
            unique_drugs.add(entry['drug2_id'])

        print(f"Precomputing graphs for {len(unique_drugs)} unique drugs...")
        failed_count = 0

        for drug_id in unique_drugs:
            smiles = self.drug_smiles.get(drug_id, '')
            if smiles:
                graph = MolecularGraph.from_smiles(smiles, drug_id, self.max_atoms)
                if graph is not None:
                    self.graph_cache[drug_id] = graph
                else:
                    failed_count += 1

        if failed_count > 0:
            print(f"Warning: Failed to create graphs for {failed_count} drugs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Returns:
            dict: Contains graph1, graph2, label, drug_ids, and idx
        """
        entry = self.data[idx]

        drug1_id = entry['drug1_id']
        drug2_id = entry['drug2_id']
        label = entry['label']

        # Get graphs from cache or compute on the fly
        if self.precompute_graphs and drug1_id in self.graph_cache and drug2_id in self.graph_cache:
            graph1 = self.graph_cache[drug1_id]
            graph2 = self.graph_cache[drug2_id]
        else:
            # Compute on the fly - memory efficient
            smiles1 = self.drug_smiles.get(drug1_id, '')
            smiles2 = self.drug_smiles.get(drug2_id, '')
            graph1 = MolecularGraph.from_smiles(smiles1, drug1_id, self.max_atoms)
            graph2 = MolecularGraph.from_smiles(smiles2, drug2_id, self.max_atoms)

        return {
            'graph1': graph1,
            'graph2': graph2,
            'label': label,
            'drug1_id': drug1_id,
            'drug2_id': drug2_id,
            'idx': idx
        }

    def get_drug_graph(self, drug_id: str) -> Optional[MolecularGraph]:
        """
        Get graph for a specific drug.

        Args:
            drug_id: DrugBank ID of the drug

        Returns:
            MolecularGraph object or None if not found
        """
        if self.precompute_graphs:
            return self.graph_cache.get(drug_id)
        else:
            smiles = self.drug_smiles.get(drug_id, '')
            return MolecularGraph.from_smiles(smiles, drug_id, self.max_atoms)


class DrugBankDataset(Dataset):
    """
    Dataset for loading raw DrugBank data with memory efficiency.
    """

    def __init__(
            self,
            data: List[Dict],
            drug_smiles: Dict[str, str],
            max_atoms: int = 100,
            transform: Optional[callable] = None
    ):
        self.data = data
        self.drug_smiles = drug_smiles
        self.max_atoms = max_atoms
        self.transform = transform
        self.graph_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        drug1_id = entry['drug1_id']
        drug2_id = entry['drug2_id']
        label = entry['label']

        # Get SMILES
        drug1_smiles = self.drug_smiles.get(drug1_id, '')
        drug2_smiles = self.drug_smiles.get(drug2_id, '')

        # Create molecular graphs
        graph1 = MolecularGraph.from_smiles(drug1_smiles, drug1_id, self.max_atoms)
        graph2 = MolecularGraph.from_smiles(drug2_smiles, drug2_id, self.max_atoms)

        if graph1 is None or graph2 is None:
            return {
                'graph1': None,
                'graph2': None,
                'label': label,
                'drug1_id': drug1_id,
                'drug2_id': drug2_id
            }

        # Apply transforms if any
        if self.transform:
            graph1 = self.transform(graph1)
            graph2 = self.transform(graph2)

        return {
            'graph1': graph1,
            'graph2': graph2,
            'label': label,
            'drug1_id': drug1_id,
            'drug2_id': drug2_id
        }


def create_dataloaders(
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
        drug_smiles: Dict[str, str],
        batch_size: int = 8,
        num_workers: int = 0,
        max_atoms: int = 100,
        max_samples: int = 5000  # Limit dataset size to prevent memory overflow
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create memory-efficient data loaders for training, validation, and testing.

    Args:
        train_data: List of training samples
        val_data: List of validation samples
        test_data: List of test samples
        drug_smiles: Dictionary mapping drug_id to SMILES string
        batch_size: Batch size for training
        num_workers: Number of workers for data loading (0 to save memory)
        max_atoms: Maximum number of atoms per molecule
        max_samples: Maximum number of samples to load (for memory efficiency)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    # Filter out invalid entries and limit dataset size
    def filter_valid(data, max_samples=None):
        filtered = []
        for entry in data:
            drug1 = entry.get('drug1_id')
            drug2 = entry.get('drug2_id')
            if drug1 in drug_smiles and drug2 in drug_smiles:
                filtered.append(entry)
            if max_samples and len(filtered) >= max_samples:
                break
        return filtered

    # Limit dataset size to prevent memory overflow
    train_data = filter_valid(train_data, max_samples=max_samples)
    val_data = filter_valid(val_data, max_samples=max_samples // 2)
    test_data = filter_valid(test_data, max_samples=max_samples // 2)

    print(f"Using {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")

    # Create datasets with precompute_graphs=False to save memory
    train_dataset = DDIDataset(train_data, drug_smiles, max_atoms, precompute_graphs=False)
    val_dataset = DDIDataset(val_data, drug_smiles, max_atoms, precompute_graphs=False)
    test_dataset = DDIDataset(test_data, drug_smiles, max_atoms, precompute_graphs=False)

    # DataLoader arguments
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'collate_fn': graph_collate_fn,
        'pin_memory': False,  # Disable pin_memory to save RAM
        'drop_last': True,
    }

    # Only add prefetch_factor if num_workers > 0
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 1

    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # Validation loader (no shuffle)
    val_loader_kwargs = loader_kwargs.copy()
    val_loader_kwargs['shuffle'] = False
    val_loader_kwargs['drop_last'] = False
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    # Test loader (no shuffle)
    test_loader_kwargs = loader_kwargs.copy()
    test_loader_kwargs['shuffle'] = False
    test_loader_kwargs['drop_last'] = False
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)

    return train_loader, val_loader, test_loader


def create_evaluation_dataloader(
        data: List[Dict],
        drug_smiles: Dict[str, str],
        batch_size: int = 8,
        num_workers: int = 0,
        max_atoms: int = 80,
        precompute_graphs: bool = False
) -> DataLoader:
    """
    Create a single dataloader for evaluation.

    Args:
        data: List of samples
        drug_smiles: Dictionary mapping drug_id to SMILES string
        batch_size: Batch size
        num_workers: Number of workers (0 to save memory)
        max_atoms: Maximum number of atoms per molecule
        precompute_graphs: Whether to precompute graphs

    Returns:
        DataLoader for evaluation
    """

    dataset = DDIDataset(data, drug_smiles, max_atoms, precompute_graphs)

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'collate_fn': graph_collate_fn,
        'pin_memory': False,
        'drop_last': False,
    }

    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 1

    return DataLoader(dataset, **loader_kwargs)


def create_small_dataset(
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
        drug_smiles: Dict[str, str],
        sample_size: int = 1000
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create a small subset of the dataset for testing/debugging.

    Args:
        train_data: Full training data
        val_data: Full validation data
        test_data: Full test data
        drug_smiles: Dictionary mapping drug_id to SMILES string
        sample_size: Number of samples to keep from each split

    Returns:
        Tuple of (small_train, small_val, small_test)
    """

    import random
    random.seed(42)

    def sample_data(data, size):
        if len(data) <= size:
            return data
        return random.sample(data, size)

    small_train = sample_data(train_data, sample_size)
    small_val = sample_data(val_data, sample_size // 2)
    small_test = sample_data(test_data, sample_size // 2)

    print(f"Created small dataset: {len(small_train)} train, {len(small_val)} val, {len(small_test)} test")

    return small_train, small_val, small_test