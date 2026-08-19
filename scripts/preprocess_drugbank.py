# scripts/preprocess_drugbank.py

# !/usr/bin/env python
"""
Preprocess DrugBank data for GANNDDI
"""

import os
import sys
import argparse
import pandas as pd
import json
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import get_default_config
from data.preprocessing import  Preprocessor
from data.drugbank_loader import DrugBankLoader
from utils.data_utils import DataUtils


def main():
    parser = argparse.ArgumentParser(description='Preprocess DrugBank data')
    parser.add_argument('--data_dir', type=str, default='data/drugbank/',
                        help='Directory containing DrugBank data')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed/',
                        help='Output directory for preprocessed data')
    parser.add_argument('--max_atoms', type=int, default=150,
                        help='Maximum number of atoms per molecule')
    parser.add_argument('--max_mol_weight', type=float, default=2000.0,
                        help='Maximum molecular weight')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test split ratio')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Load config
    config = get_default_config()

    # Update config with args
    config_dict = {
        'model': config.model,
        'data': {
            'drugbank_path': args.data_dir,
            'max_atoms': args.max_atoms,
            'max_mol_weight': args.max_mol_weight,
        },
        'training': config.training,
        'evaluation': {
            'test_split': args.test_split,
            'val_split': args.val_split,
        }
    }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("GANNDDI - DrugBank Data Preprocessing")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max atoms: {args.max_atoms}")
    print(f"Max molecular weight: {args.max_mol_weight}")
    print(f"Test split: {args.test_split}")
    print(f"Validation split: {args.val_split}")
    print("=" * 60)

    # Check if data exists
    required_files = [
        'drug_links.csv',
        'drugbank_vocabulary.csv',
        'drugbank_all_structures.sdf',
        'ddi_data.csv'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.data_dir, file)):
            missing_files.append(file)

    if missing_files:
        print("\nWarning: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run scripts/download_data.sh first")
        print("Make sure ddi_data.csv is in the data directory")

        # Check if we can continue with partial data
        if 'ddi_data.csv' in missing_files:
            print("\nERROR: ddi_data.csv is required for DDI prediction")
            print("Please obtain DDI data from DrugBank or other sources")
            sys.exit(1)

    # Initialize preprocessor
    preprocessor = Preprocessor(config_dict)

    # Load and preprocess data
    print("\nLoading DrugBank data...")
    loader = DrugBankLoader(config_dict)
    data = loader.load_all_data()

    print("\nPreprocessing complete!")
    print(f"Training samples: {len(data['train_data'])}")
    print(f"Validation samples: {len(data['val_data'])}")
    print(f"Test samples: {len(data['test_data'])}")
    print(f"Number of drugs: {len(data['drug_smiles'])}")
    print(f"Number of interaction types: {len(data['label_map'])}")

    # Save preprocessed data
    print(f"\nSaving preprocessed data to {args.output_dir}...")

    # Save datasets
    pd.DataFrame(data['train_data']).to_csv(
        os.path.join(args.output_dir, 'train.csv'), index=False
    )
    pd.DataFrame(data['val_data']).to_csv(
        os.path.join(args.output_dir, 'val.csv'), index=False
    )
    pd.DataFrame(data['test_data']).to_csv(
        os.path.join(args.output_dir, 'test.csv'), index=False
    )

    # Save drug smiles
    with open(os.path.join(args.output_dir, 'drug_smiles.json'), 'w') as f:
        json.dump(data['drug_smiles'], f, indent=2)

    # Save drug info
    with open(os.path.join(args.output_dir, 'drug_info.json'), 'w') as f:
        json.dump(data['drug_info'], f, indent=2)

    # Save label map
    with open(os.path.join(args.output_dir, 'label_map.json'), 'w') as f:
        json.dump(data['label_map'], f, indent=2)

    # Save statistics
    stats = {
        'num_train': len(data['train_data']),
        'num_val': len(data['val_data']),
        'num_test': len(data['test_data']),
        'num_drugs': len(data['drug_smiles']),
        'num_interaction_types': len(data['label_map']),
        'interaction_types': list(data['label_map'].keys()),
        'max_atoms': args.max_atoms,
        'max_mol_weight': args.max_mol_weight,
        'test_split': args.test_split,
        'val_split': args.val_split,
        'seed': args.seed
    }

    with open(os.path.join(args.output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Print dataset statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Training samples: {stats['num_train']:,}")
    print(f"Validation samples: {stats['num_val']:,}")
    print(f"Test samples: {stats['num_test']:,}")
    print(f"Total drugs: {stats['num_drugs']:,}")
    print(f"Interaction types: {stats['num_interaction_types']}")

    # Print class distribution
    print("\nClass distribution (top 10):")
    train_labels = [item['label'] for item in data['train_data']]
    label_counts = pd.Series(train_labels).value_counts()

    # Map labels to names
    reverse_label_map = {v: k for k, v in data['label_map'].items()}
    for label, count in label_counts.head(10).items():
        label_name = reverse_label_map.get(label, f'Class_{label}')
        print(f"  {label_name}: {count:,} ({count / len(train_labels) * 100:.1f}%)")

    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
