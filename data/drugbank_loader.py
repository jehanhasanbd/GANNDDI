# data/drugbank_loader.py

"""
DrugBank data loader
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict, List, Tuple, Optional
import os
import json
from tqdm import tqdm

from data.preprocessing import Preprocessor
from data.molecular_graph import MolecularGraph


class DrugBankLoader:
    """Load and preprocess DrugBank data"""

    def __init__(self, config):
        self.config = config
        self.preprocessor = Preprocessor(config)

        # Get data paths from config
        data_config = config.get('data', {})
        self.data_path = data_config.get('drugbank_path', 'data/drugbank/')

        # If data_path is preprocessed, try to find raw data
        if 'preprocessed' in self.data_path:
            self.raw_data_path = 'data/drugbank/'
        else:
            self.raw_data_path = self.data_path

        self.drug_smiles = {}
        self.drug_info = {}
        self.ddi_pairs = []
        self.label_map = {}
        self.dataset = []

    def load_all_data(self) -> Dict:
        """Load all DrugBank data"""

        # Check if preprocessed data exists
        preprocessed_dir = os.path.join(self.data_path, 'preprocessed')
        if not os.path.exists(preprocessed_dir):
            preprocessed_dir = self.data_path

        # Check if preprocessed data exists
        if os.path.exists(os.path.join(preprocessed_dir, 'train.csv')):
            print("Loading preprocessed data...")
            return self.preprocessor.load_preprocessed_data(preprocessed_dir)

        # Load raw data
        print("Loading raw DrugBank data...")

        # Check if raw data exists
        if not os.path.exists(self.raw_data_path):
            print(f"Warning: Raw data directory {self.raw_data_path} not found")
            print("Creating synthetic data for testing...")
            return self._create_synthetic_data()

        # Load drug links
        drug_links_path = os.path.join(self.raw_data_path, 'drug_links.csv')
        if not os.path.exists(drug_links_path):
            print(f"Warning: {drug_links_path} not found")
            drug_links_df = pd.DataFrame()
        else:
            drug_links_df = self.preprocessor.load_drug_data(drug_links_path)

        # Load structures
        structures_path = os.path.join(self.raw_data_path, 'drugbank_all_structures.sdf')
        if os.path.exists(structures_path):
            structures_df = self.preprocessor.load_structures(structures_path)
        else:
            print(f"Warning: {structures_path} not found")
            structures_df = pd.DataFrame()

        # Load DDI data
        ddi_path = os.path.join(self.raw_data_path, 'ddi_data.csv')
        if os.path.exists(ddi_path):
            ddi_data = self.preprocessor.load_ddi_data(ddi_path)
        else:
            print(f"Warning: {ddi_path} not found")
            print("Creating synthetic DDI data...")
            ddi_data = pd.DataFrame()

        # Preprocess drugs
        drug_smiles, drug_info = self.preprocessor.preprocess_drugs(
            drug_links_df, structures_df
        )

        # If no drugs found, create synthetic data
        if len(drug_smiles) == 0:
            print("No drugs found. Creating synthetic data...")
            return self._create_synthetic_data()

        # Build DDI pairs
        ddi_pairs, label_map = self.preprocessor.build_ddi_pairs(
            drug_smiles, ddi_data
        )

        # If no DDI pairs found, create synthetic pairs
        if len(ddi_pairs) == 0:
            print("No DDI pairs found. Creating synthetic pairs...")
            ddi_pairs, label_map = self.preprocessor._create_synthetic_ddi_pairs(drug_smiles)

            # After creating dataset, limit the size
            dataset = self.preprocessor.create_dataset(ddi_pairs, drug_smiles, drug_info)

            # Filter dataset
            max_atoms = self.config.get('data', {}).get('max_atoms', 100)
            max_mol_weight = self.config.get('data', {}).get('max_mol_weight', 1000)
            dataset = self.preprocessor.filter_dataset(dataset, max_atoms, max_mol_weight)

            # LIMIT DATASET SIZE - take only first N samples
            max_samples = self.config.get('data', {}).get('max_samples', 5000)
            if len(dataset) > max_samples:
                print(f"Limiting dataset from {len(dataset)} to {max_samples} samples")
                dataset = dataset[:max_samples]

            # Split dataset
            test_split = self.config.get('evaluation', {}).get('test_split', 0.2)
            val_split = self.config.get('evaluation', {}).get('val_split', 0.1)
            train, val, test = self.preprocessor.split_dataset(
                dataset, test_split, val_split
            )

        # Save preprocessed data
        preprocessed_dir = os.path.join(self.raw_data_path, 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)
        self.preprocessor.save_preprocessed_data(
            preprocessed_dir,
            train, val, test,
            drug_smiles, drug_info, label_map
        )

        return {
            'train_data': train,
            'val_data': val,
            'test_data': test,
            'drug_smiles': drug_smiles,
            'drug_info': drug_info,
            'label_map': label_map
        }

    def _create_synthetic_data(self) -> Dict:
        """Create synthetic data for testing"""

        print("Creating synthetic data...")

        # Create synthetic drug SMILES
        synthetic_smiles = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CC(=O)NC1=CC=C(C=C1)O',  # Paracetamol
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC1=CC=C(C=C1)CC(C)N',  # Amphetamine
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Theophylline
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Naproxen
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin (duplicate)
        ]

        drug_ids = [f"DB{i:05d}" for i in range(1, len(synthetic_smiles) + 1)]

        drug_smiles = {}
        drug_info = {}
        for drug_id, smiles in zip(drug_ids, synthetic_smiles):
            drug_smiles[drug_id] = smiles
            drug_info[drug_id] = {
                'name': f'Drug_{drug_id}',
                'smiles': smiles,
                'formula': '',
                'molecular_weight': 0.0,
                'drug_groups': '',
                'synonyms': ''
            }

        # Create synthetic DDI pairs
        import random
        random.seed(42)

        ddi_pairs = []
        label_map = {'interaction': 0, 'no_interaction': 1}
        used_pairs = set()

        # Create positive pairs
        for i in range(len(drug_ids)):
            for j in range(i + 1, len(drug_ids)):
                if random.random() < 0.3:  # 30% chance of interaction
                    drug1 = drug_ids[i]
                    drug2 = drug_ids[j]
                    pair = tuple(sorted([drug1, drug2]))
                    if pair not in used_pairs:
                        used_pairs.add(pair)
                        ddi_pairs.append((drug1, drug2, 0))

        # Create negative pairs
        for i in range(len(drug_ids)):
            for j in range(i + 1, len(drug_ids)):
                drug1 = drug_ids[i]
                drug2 = drug_ids[j]
                pair = tuple(sorted([drug1, drug2]))
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    ddi_pairs.append((drug1, drug2, 1))

        # Create dataset
        dataset = self.preprocessor.create_dataset(
            ddi_pairs, drug_smiles, drug_info
        )

        # Split dataset
        test_split = 0.2
        val_split = 0.1
        train, val, test = self.preprocessor.split_dataset(
            dataset, test_split, val_split
        )

        # Save preprocessed data
        preprocessed_dir = os.path.join(self.data_path, 'preprocessed')
        if not os.path.exists(preprocessed_dir):
            preprocessed_dir = 'data/preprocessed/'
        os.makedirs(preprocessed_dir, exist_ok=True)

        self.preprocessor.save_preprocessed_data(
            preprocessed_dir,
            train, val, test,
            drug_smiles, drug_info, label_map
        )

        return {
            'train_data': train,
            'val_data': val,
            'test_data': test,
            'drug_smiles': drug_smiles,
            'drug_info': drug_info,
            'label_map': label_map
        }

    def load_drug_smiles(self, drug_id: str) -> Optional[str]:
        """Get SMILES for a drug ID"""
        if not self.drug_smiles:
            data = self.load_all_data()
            self.drug_smiles = data.get('drug_smiles', {})

        return self.drug_smiles.get(drug_id)

    def load_drug_info(self, drug_id: str) -> Optional[Dict]:
        """Get drug information for a drug ID"""
        if not self.drug_info:
            data = self.load_all_data()
            self.drug_info = data.get('drug_info', {})

        return self.drug_info.get(drug_id)

    def get_ddi_entries(self, num_samples: Optional[int] = None) -> List[Dict]:
        """Get DDI entries"""
        if not self.dataset:
            data = self.load_all_data()
            self.dataset = data.get('train_data', [])
            self.dataset.extend(data.get('val_data', []))
            self.dataset.extend(data.get('test_data', []))

        if num_samples is not None:
            return self.dataset[:num_samples]
        return self.dataset

    def get_ddi_pairs(self) -> List[Tuple[str, str, int]]:
        """Get DDI pairs with labels"""
        if not self.ddi_pairs:
            entries = self.get_ddi_entries()
            self.ddi_pairs = [
                (entry['drug1_id'], entry['drug2_id'], entry['label'])
                for entry in entries
            ]

        return self.ddi_pairs