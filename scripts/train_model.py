# scripts/train_model.py

# !/usr/bin/env python
"""
Train GANNDDI model
"""

import os
import sys
import argparse
import torch
import json
import yaml
from datetime import datetime
import random
import numpy as np

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports
from config.model_config import get_default_config, ModelConfig
from data.dataset import  create_dataloaders
from data.drugbank_loader import DrugBankLoader
from models.ddi_predictor import DDIPredictor
from training.trainer import Trainer


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train GANNDDI model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed/',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='checkpoints/',
                        help='Output directory for model checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--num_patterns', type=int, default=None,
                        help='Number of SIE patterns')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    parser.add_argument('--wandb', action='store_true',
                        help='Use wandb logging')
    parser.add_argument('--wandb_project', type=str, default='gannddi',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Load config
    if args.config and os.path.exists(args.config):
        print(f"Loading config from: {args.config}")
        config = ModelConfig.from_yaml(args.config)
    else:
        print("Using default config")
        config = get_default_config()

    # Convert to dict
    config_dict = {
        'model': config.model,
        'data': config.data,
        'training': config.training,
        'evaluation': config.evaluation
    }

    # Override training parameters
    if args.epochs:
        config_dict['training']['epochs'] = args.epochs
    if args.batch_size:
        config_dict['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config_dict['training']['learning_rate'] = args.learning_rate

    # Override model parameters
    if args.hidden_dim:
        config_dict['model']['gaan']['hidden_dim'] = args.hidden_dim
        config_dict['model']['gate_encoder']['hidden_dim'] = args.hidden_dim
        config_dict['model']['sie_encoder']['hidden_dim'] = args.hidden_dim
        config_dict['model']['ddi_predictor']['hidden_dim'] = args.hidden_dim
    if args.num_heads:
        config_dict['model']['gaan']['num_heads'] = args.num_heads
        config_dict['model']['gate_encoder']['num_heads'] = args.num_heads
    if args.num_patterns:
        config_dict['model']['sie_encoder']['num_patterns'] = args.num_patterns
    if args.dropout:
        config_dict['model']['gaan']['dropout'] = args.dropout
        config_dict['model']['gate_encoder']['dropout'] = args.dropout
        config_dict['model']['sie_encoder']['dropout'] = args.dropout
        config_dict['model']['ddi_predictor']['dropout'] = args.dropout

    # Update data path
    if os.path.exists(args.data_dir):
        config_dict['data']['drugbank_path'] = args.data_dir
    else:
        print(f"Warning: Data directory {args.data_dir} not found")

    # Print configuration
    print("\n" + "=" * 60)
    print("GANNDDI Training Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {config_dict['training']['epochs']}")
    print(f"Batch size: {config_dict['data']['batch_size']}")
    print(f"Learning rate: {config_dict['training']['learning_rate']}")
    print(f"Hidden dimension: {config_dict['model']['gaan']['hidden_dim']}")
    print(f"Number of heads: {config_dict['model']['gaan']['num_heads']}")
    print(f"Number of patterns: {config_dict['model']['sie_encoder']['num_patterns']}")
    print(f"Dropout: {config_dict['model']['ddi_predictor']['dropout']}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load data
    print("\nLoading preprocessed data...")
    loader = DrugBankLoader(config_dict)
    data = loader.load_all_data()

    # Check if data was loaded successfully
    if not data or 'train_data' not in data or len(data['train_data']) == 0:
        print("ERROR: No data loaded. Please run preprocess_drugbank.py first.")
        sys.exit(1)

    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    drug_smiles = data['drug_smiles']
    label_map = data['label_map']

    print(f"Training samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    print(f"Test samples: {len(test_data):,}")
    print(f"Number of drugs: {len(drug_smiles):,}")
    print(f"Number of interaction types: {len(label_map)}")

    # Create dataloaders
    data_config = config.get_data_config()
    batch_size = data_config.batch_size
    num_workers = data_config.num_workers
    max_atoms = data_config.max_atoms

    print(f"\nCreating dataloaders (batch_size={batch_size})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        drug_smiles, batch_size, num_workers, max_atoms
    )

    # Initialize model
    print("\nInitializing model...")
    num_classes = len(label_map)

    model_config = config.get_ddi_predictor_config()
    model = DDIPredictor(
        node_feat_dim=74,
        edge_feat_dim=10,
        hidden_dim=model_config.hidden_dim,
        num_patterns=config.model.get('sie_encoder', {}).get('num_patterns', 16),
        num_heads=config.model.get('gaan', {}).get('num_heads', 8),
        dropout=model_config.dropout,
        num_classes=num_classes
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Initialize trainer
    print("\nInitializing trainer...")

    # Initialize wandb if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"gannddi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config_dict
            )
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")

    trainer = Trainer(
        model=model,
        config=config_dict,
        device=device,
        use_wandb=args.wandb
    )

    # Train
    print("\nStarting training...")
    print("-" * 60)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=args.output_dir
    )

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if history.get('test'):
        test_metrics = history['test']
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        print(f"Test AUROC: {test_metrics.get('auroc', 0.0):.4f}")
        print(f"Test AUPRC: {test_metrics.get('auprc', 0.0):.4f}")
    else:
        print("No test results available")

    # Print best validation results
    best_val = history.get('best_val', {})
    if best_val:
        print(f"\nBest Validation Accuracy: {best_val.get('accuracy', 0.0):.4f}")
        print(f"Best Validation F1: {best_val.get('f1', 0.0):.4f}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': trainer.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'config': config_dict,
        'test_metrics': history.get('test', {}),
        'best_val_metrics': best_val,
        'num_classes': num_classes
    }, final_model_path)

    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")

    # Save label map for inference
    with open(os.path.join(args.output_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

    print("\nDone!")


if __name__ == '__main__':
    main()