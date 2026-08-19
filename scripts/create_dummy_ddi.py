# scripts/create_dummy_ddi.py

# !/usr/bin/env python
"""
Create dummy DDI data for testing
"""

import pandas as pd
import random
import os
from rdkit import Chem
from rdkit.Chem import Descriptors


def create_dummy_ddi_data(output_path='data/drugbank/ddi_data.csv', num_pairs=1000):
    """Create dummy DDI data for testing"""

    # Sample drugs from DrugBank
    drugbank_vocab_path = 'data/drugbank/drugbank_vocabulary.csv'

    if not os.path.exists(drugbank_vocab_path):
        print(f"Error: {drugbank_vocab_path} not found")
        print("Please run download_data.sh first")
        return

    # Load drug vocabulary
    df_vocab = pd.read_csv(drugbank_vocab_path)

    # Get drug IDs
    drug_ids = df_vocab['DrugBank ID'].dropna().tolist()

    if len(drug_ids) < 10:
        print("Error: Not enough drug IDs found")
        return

    # Interaction types
    interaction_types = [
        'synergistic', 'antagonistic', 'additive', 'potentiation',
        'inhibition', 'activation', 'metabolism', 'transport',
        'binding', 'cleavage', 'degradation', 'induction',
        'suppression', 'enhancement', 'reduction', 'modulation'
    ]

    # Generate random pairs
    data = []
    used_pairs = set()

    for _ in range(num_pairs):
        drug1 = random.choice(drug_ids)
        drug2 = random.choice(drug_ids)

        # Avoid self-pairs and duplicates
        pair = tuple(sorted([drug1, drug2]))
        if pair[0] == pair[1] or pair in used_pairs:
            continue

        used_pairs.add(pair)

        # Random interaction
        interaction = random.choice(interaction_types)

        # Random confidence
        confidence = random.uniform(0.5, 1.0)

        data.append({
            'drug1_id': pair[0],
            'drug2_id': pair[1],
            'interaction_type': interaction,
            'confidence': confidence,
            'source': 'dummy'
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Created {len(df)} dummy DDI pairs")
    print(f"Saved to: {output_path}")

    # Print statistics
    print(f"\nInteraction type distribution:")
    print(df['interaction_type'].value_counts())

    print(f"\nSample data:")
    print(df.head(10))


if __name__ == '__main__':
    create_dummy_ddi_data()