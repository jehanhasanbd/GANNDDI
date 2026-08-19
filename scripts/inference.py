# scripts/inference.py

# !/usr/bin/env python
"""
Inference script for GANNDDI model
"""

import os
import sys
import argparse
import torch
import json
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ddi_predictor import DDIPredictor
from data.molecular_graph import MolecularGraph
from utils.chem_utils import ChemistryUtils


def load_model(model_path, device='cuda'):
    """Load trained model"""

    checkpoint = torch.load(model_path, map_location=device)

    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    num_classes = checkpoint.get('num_classes', 65)

    # Initialize model
    model = DDIPredictor(
        node_feat_dim=74,
        edge_feat_dim=10,
        hidden_dim=model_config.get('ddi_predictor', {}).get('hidden_dim', 256),
        num_patterns=model_config.get('sie_encoder', {}).get('num_patterns', 16),
        num_heads=model_config.get('gaan', {}).get('num_heads', 8),
        dropout=model_config.get('ddi_predictor', {}).get('dropout', 0.2),
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load label map if available
    label_map = {}
    label_map_path = os.path.join(os.path.dirname(model_path), 'label_map.json')
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)

    return model, label_map, config


def predict_single_pair(model, smiles1, smiles2, device='cuda', max_atoms=150):
    """Predict interaction for a single drug pair"""

    # Create molecular graphs
    graph1 = MolecularGraph.from_smiles(smiles1, 'drug1', max_atoms)
    graph2 = MolecularGraph.from_smiles(smiles2, 'drug2', max_atoms)

    if graph1 is None or graph2 is None:
        return None, None

    # Prepare batch
    batch = {
        'node_features1': graph1.node_features.unsqueeze(0).to(device),
        'edge_index1': graph1.edge_index.unsqueeze(0).to(device),
        'edge_features1': graph1.edge_features.unsqueeze(0).to(device),
        'node_features2': graph2.node_features.unsqueeze(0).to(device),
        'edge_index2': graph2.edge_index.unsqueeze(0).to(device),
        'edge_features2': graph2.edge_features.unsqueeze(0).to(device),
    }

    # Predict
    with torch.no_grad():
        logits = model(
            batch['node_features1'],
            batch['edge_index1'],
            batch['edge_features1'],
            batch['node_features2'],
            batch['edge_index2'],
            batch['edge_features2']
        )
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy(), logits.cpu().numpy()


def predict_batch(model, pairs, device='cuda', max_atoms=150, batch_size=32):
    """Predict interactions for multiple drug pairs"""

    all_probs = []
    all_logits = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="Predicting"):
        batch_pairs = pairs[i:i + batch_size]
        batch_probs = []
        batch_logits = []

        for smiles1, smiles2 in batch_pairs:
            probs, logits = predict_single_pair(model, smiles1, smiles2, device, max_atoms)
            if probs is not None:
                batch_probs.append(probs[0])
                batch_logits.append(logits[0])
            else:
                batch_probs.append([0.0])
                batch_logits.append([0.0])

        all_probs.extend(batch_probs)
        all_logits.extend(batch_logits)

    return all_probs, all_logits


def main():
    parser = argparse.ArgumentParser(description='GANNDDI Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file with drug pairs (CSV) or SMILES pair (format: SMILES1,SMILES2)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output file for predictions')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.model_path}")
    model, label_map, config = load_model(args.model_path, device)

    num_classes = model.classifier[-1].out_features
    print(f"Model loaded. {num_classes} interaction types.")

    # Process input
    if os.path.exists(args.input):
        # Read from CSV file
        print(f"Reading input from: {args.input}")
        df = pd.read_csv(args.input)

        # Check required columns
        if 'SMILES1' not in df.columns or 'SMILES2' not in df.columns:
            print("ERROR: CSV must contain 'SMILES1' and 'SMILES2' columns")
            sys.exit(1)

        pairs = list(zip(df['SMILES1'].values, df['SMILES2'].values))
    else:
        # Single pair from command line
        try:
            smiles1, smiles2 = args.input.split(',')
            pairs = [(smiles1.strip(), smiles2.strip())]
        except:
            print("ERROR: Input must be either a CSV file or 'SMILES1,SMILES2'")
            sys.exit(1)

    print(f"Processing {len(pairs)} drug pairs...")

    # Predict
    probs, logits = predict_batch(model, pairs, device, batch_size=args.batch_size)

    # Prepare results
    results = []
    for i, (smiles1, smiles2) in enumerate(pairs):
        result = {
            'SMILES1': smiles1,
            'SMILES2': smiles2,
        }

        # Add probabilities for each class
        if i < len(probs):
            for j, prob in enumerate(probs[i]):
                label_name = label_map.get(str(j), f'Class_{j}')
                result[f'prob_{label_name}'] = prob

            # Add predicted class
            pred_class = np.argmax(probs[i])
            pred_label = label_map.get(str(pred_class), f'Class_{pred_class}')
            result['predicted_class'] = pred_class
            result['predicted_label'] = pred_label
            result['confidence'] = np.max(probs[i])

        results.append(result)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)

    print(f"Predictions saved to: {args.output}")

    # Print summary
    print("\nPrediction Summary:")
    print(f"Total pairs processed: {len(pairs)}")
    print(f"Number of classes: {num_classes}")

    # Show sample predictions
    print("\nSample predictions:")
    print(df_results.head(10).to_string())


if __name__ == '__main__':
    import numpy as np

    main()