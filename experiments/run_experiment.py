# experiments/run_experiment.py

import torch
import os
import yaml
import json
import argparse
from datetime import datetime
import wandb

from config.model_config import get_default_config, ModelConfig
from data.drugbank_loader import DrugBankLoader
from data.dataset import create_dataloaders
from models.ddi_predictor import DDIPredictor
from training.trainer import Trainer


def run_experiment(config_path: str = None, use_wandb: bool = False):
    """Run a full experiment"""

    # Load config
    if config_path:
        config = ModelConfig.from_yaml(config_path)
    else:
        config = get_default_config()

    # Convert to dict
    config_dict = {
        'model': config.model,
        'data': config.data,
        'training': config.training,
        'evaluation': config.evaluation
    }

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="gannddi",
            config=config_dict,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Load data
    print("Loading DrugBank data...")
    loader = DrugBankLoader(config_dict)
    data = loader.load_all_data()

    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    drug_smiles = data['drug_smiles']
    label_map = data['label_map']

    # Create dataloaders
    data_config = config.get_data_config()
    batch_size = data_config.batch_size
    num_workers = data_config.num_workers
    max_atoms = data_config.max_atoms

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        drug_smiles, batch_size, num_workers, max_atoms
    )

    # Initialize model
    print("Initializing model...")
    num_classes = len(label_map)

    model_config = config.get_ddi_predictor_config()
    model = DDIPredictor(
        node_feat_dim=74,
        edge_feat_dim=10,
        hidden_dim=model_config.hidden_dim,
        num_heads=config.model.get('gaan', {}).get('num_heads', 8),
        num_patterns=config.model.get('sie_encoder', {}).get('num_patterns', 16),
        dropout=model_config.dropout,
        num_classes=num_classes
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=config_dict,
        use_wandb=use_wandb
    )

    # Train
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir='checkpoints/'
    )

    # Print final results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

    if history.get('test'):
        test_metrics = history['test']
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test AUROC: {test_metrics.get('auroc', 0.0):.4f}")

    if use_wandb:
        wandb.finish()

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    args = parser.parse_args()

    run_experiment(args.config, args.wandb)