#!/usr/bin/env python
"""
Evaluate trained GANNDDI model
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports from gannddi package
from models.ddi_predictor import DDIPredictor
from data.dataset import DDIDataset, graph_collate_fn
from training.evaluator import Evaluator
from utils.metrics import MetricsCalculator


def create_evaluation_dataloader(
        data: List[Dict],
        drug_smiles: Dict[str, str],
        batch_size: int = 8,
        num_workers: int = 0,
        max_atoms: int = 80,
        precompute_graphs: bool = False
):
    """Create a single dataloader for evaluation"""

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


def main():
    parser = argparse.ArgumentParser(description='Evaluate GANNDDI model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed/',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='evaluation/',
                        help='Output directory for evaluation results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index (use -1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to evaluate (for memory)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--save_confusion', action='store_true',
                        help='Save confusion matrix plot')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    args = parser.parse_args()

    # Set device
    if args.no_cuda or args.gpu < 0:
        device = 'cpu'
        print("Using CPU")
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Get model config
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Get num_classes from checkpoint
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    elif 'classifier.8.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['classifier.8.weight'].shape[0]
    else:
        try:
            with open(os.path.join(args.data_dir, 'label_map.json'), 'r') as f:
                label_map = json.load(f)
                num_classes = len(label_map)
        except:
            num_classes = 2

    print(f"Number of classes from checkpoint: {num_classes}")

    # Initialize model with correct num_classes
    model = DDIPredictor(
        node_feat_dim=74,
        edge_feat_dim=10,
        hidden_dim=model_config.get('ddi_predictor', {}).get('hidden_dim', 64),
        num_patterns=model_config.get('sie_encoder', {}).get('num_patterns', 4),
        num_heads=model_config.get('gaan', {}).get('num_heads', 4),
        dropout=model_config.get('ddi_predictor', {}).get('dropout', 0.3),
        num_classes=num_classes
    )

    # Load state dict with strict=False
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model = model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load label map
    try:
        with open(os.path.join(args.data_dir, 'label_map.json'), 'r') as f:
            label_map = json.load(f)
        print(f"Loaded label map with {len(label_map)} classes")
    except:
        if num_classes == 1:
            label_map = {'interaction': 0}
        elif num_classes == 2:
            label_map = {'interaction': 0, 'no_interaction': 1}
        else:
            label_map = {str(i): f'Class_{i}' for i in range(num_classes)}
        print(f"Using default label map with {len(label_map)} classes")

    # Load data
    print(f"Loading data from: {args.data_dir}")

    try:
        with open(os.path.join(args.data_dir, 'drug_smiles.json'), 'r') as f:
            drug_smiles = json.load(f)
        print(f"Loaded {len(drug_smiles)} drugs")
    except:
        print("Warning: drug_smiles.json not found, using empty dict")
        drug_smiles = {}

    # Load dataset
    split_map = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv'
    }

    data_path = os.path.join(args.data_dir, split_map[args.split])
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        print("Please run preprocess_drugbank.py first")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {args.split} set")

    # Filter labels to match num_classes
    if 'label' in df.columns:
        df = df[df['label'] < num_classes]
        print(f"Filtered to {len(df)} samples with valid labels")

    # Limit samples
    if len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
        print(f"Limited to {args.max_samples} samples")

    # Convert to list of dicts
    data = df.to_dict('records')

    if len(data) == 0:
        print("Error: No samples available for evaluation")
        sys.exit(1)

    # Create dataloader directly
    print(f"\nCreating dataloader with {len(data)} samples...")
    test_loader = create_evaluation_dataloader(
        data,
        drug_smiles,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_atoms=80,
        precompute_graphs=False
    )

    # Evaluate
    print("\nRunning evaluation...")
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(model, test_loader, device)

    # Print results
    print("\n" + "=" * 50)
    print(f"Evaluation Results on {args.split} set")
    print("=" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric.upper():15s}: {value:.4f}")
        else:
            print(f"{metric.upper():15s}: {value}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'metrics_{args.split}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Get detailed predictions if requested
    if args.save_predictions or args.save_confusion:
        print("\nCollecting predictions...")

        all_preds = []
        all_labels = []
        all_scores = []
        all_drug_ids = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # Skip empty batches
                if batch['labels'].numel() == 0:
                    continue

                # Move to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                # Forward pass
                logits = model(
                    batch['node_features1'],
                    batch['edge_index1'],
                    batch['edge_features1'],
                    batch['node_features2'],
                    batch['edge_index2'],
                    batch['edge_features2']
                )

                preds = torch.argmax(logits, dim=1)
                scores = torch.softmax(logits, dim=1)

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(batch['labels'].detach().cpu().numpy())
                all_scores.extend(scores.detach().cpu().numpy())

                if 'drug_ids1' in batch and 'drug_ids2' in batch:
                    for d1, d2 in zip(batch['drug_ids1'], batch['drug_ids2']):
                        all_drug_ids.append((d1, d2))

        # Create reverse label map
        reverse_label_map = {v: k for k, v in label_map.items()}

        # Save predictions
        if args.save_predictions and len(all_labels) > 0:
            predictions_df = pd.DataFrame({
                'true_label': all_labels,
                'predicted_label': all_preds,
            })

            # Add label names
            predictions_df['true_label_name'] = predictions_df['true_label'].map(
                lambda x: reverse_label_map.get(x, f'Class_{x}')
            )
            predictions_df['predicted_label_name'] = predictions_df['predicted_label'].map(
                lambda x: reverse_label_map.get(x, f'Class_{x}')
            )

            # Add scores for each class
            for i in range(num_classes):
                class_name = reverse_label_map.get(i, f'Class_{i}')
                predictions_df[f'score_{class_name}'] = [s[i] if i < len(s) else 0.0 for s in all_scores]

            # Add drug IDs if available
            if all_drug_ids:
                predictions_df['drug1_id'] = [d[0] for d in all_drug_ids]
                predictions_df['drug2_id'] = [d[1] for d in all_drug_ids]

            predictions_path = os.path.join(args.output_dir, f'predictions_{args.split}.csv')
            predictions_df.to_csv(predictions_path, index=False)
            print(f"Predictions saved to: {predictions_path}")

        # Generate classification report
        if len(all_labels) > 0:
            target_names = [reverse_label_map.get(i, f'Class_{i}') for i in range(num_classes)]
            try:
                report = classification_report(
                    all_labels, all_preds,
                    target_names=target_names,
                    zero_division=0
                )

                report_path = os.path.join(args.output_dir, f'classification_report_{args.split}.txt')
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"Classification report saved to: {report_path}")
            except Exception as e:
                print(f"Could not generate classification report: {e}")

        # Save confusion matrix plot
        if args.save_confusion and len(all_labels) > 0:
            cm = confusion_matrix(all_labels, all_preds)

            # Plot confusion matrix
            plt.figure(figsize=(max(8, num_classes * 0.5), max(6, num_classes * 0.5)))

            # Only show classes that appear in the data
            unique_labels = sorted(set(all_labels + all_preds))
            cm_subset = cm[np.ix_(unique_labels, unique_labels)]

            sns.heatmap(
                cm_subset,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[target_names[i] for i in unique_labels],
                yticklabels=[target_names[i] for i in unique_labels],
                square=True
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {args.split} set')
            plt.tight_layout()

            cm_path = os.path.join(args.output_dir, f'confusion_matrix_{args.split}.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to: {cm_path}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()