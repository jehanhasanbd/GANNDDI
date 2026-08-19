# scripts/create_ddi_from_targets.py
# !/usr/bin/env python
"""
Create DDI data from DrugBank target information
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_ddi_from_targets():
    """Create DDI pairs based on shared targets"""

    # Load target data
    targets_path = 'data/drugbank/drugbank_all_target_polypeptide_ids - all.csv'

    if not os.path.exists(targets_path):
        print(f"Error: {targets_path} not found")
        return

    # Load target data
    df_targets = pd.read_csv(targets_path)

    print(f"Loaded {len(df_targets)} target records")
    print(f"Columns: {df_targets.columns.tolist()}")

    # Check columns
    # Expected columns: ID, Name, Gene Name, UniProt ID, Drug IDs, etc.

    # If the file doesn't have the expected columns, try to infer
    if 'Drug IDs' not in df_targets.columns:
        # Try to find drug ID column
        drug_col = None
        for col in df_targets.columns:
            if 'drug' in col.lower() or 'Drug' in col:
                drug_col = col
                break

        if drug_col is None:
            print("Error: Cannot find Drug IDs column")
            print(f"Available columns: {df_targets.columns.tolist()}")
            return
    else:
        drug_col = 'Drug IDs'

    # Parse drug IDs from the column (they might be semicolon separated)
    drug_to_targets = defaultdict(set)
    target_to_drugs = defaultdict(set)

    for idx, row in df_targets.iterrows():
        drug_ids = row[drug_col]
        if pd.isna(drug_ids) or drug_ids == '':
            continue

        # Split by semicolon or comma
        if ';' in str(drug_ids):
            drugs = [d.strip() for d in str(drug_ids).split(';')]
        elif ',' in str(drug_ids):
            drugs = [d.strip() for d in str(drug_ids).split(',')]
        else:
            drugs = [str(drug_ids).strip()]

        # Get target ID
        target_id = row.get('UniProt ID', row.get('ID', row.get('Name', f'Target_{idx}')))

        for drug in drugs:
            if drug and drug != '':
                drug_to_targets[drug].add(target_id)
                target_to_drugs[target_id].add(drug)

    print(f"\nFound {len(drug_to_targets)} drugs with targets")
    print(f"Found {len(target_to_drugs)} targets")

    # Create DDI pairs based on shared targets
    ddi_pairs = []
    drugs = list(drug_to_targets.keys())

    print(f"\nCreating DDI pairs based on shared targets...")

    # Interaction types based on target relationships
    interaction_types = [
        'target_shared',
        'target_related',
        'target_interaction',
        'synergistic',
        'antagonistic',
        'additive'
    ]

    # For each pair of drugs, check if they share targets
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            drug1 = drugs[i]
            drug2 = drugs[j]

            targets1 = drug_to_targets[drug1]
            targets2 = drug_to_targets[drug2]

            # Check if they share targets
            shared_targets = targets1 & targets2

            if shared_targets:
                # They share targets - potential DDI
                interaction_type = 'target_shared'

                # Determine if related targets (different targets in same pathway)
                # For simplicity, we'll just use shared target interaction

                ddi_pairs.append({
                    'drug1_id': drug1,
                    'drug2_id': drug2,
                    'interaction_type': interaction_type,
                    'description': f'Share {len(shared_targets)} target(s)',
                    'severity': 'moderate',
                    'source': 'DrugBank_Targets',
                    'confidence': 0.7 + (len(shared_targets) * 0.05)  # More shared targets = higher confidence
                })

    print(f"Created {len(ddi_pairs)} DDI pairs based on shared targets")

    # Also create some synthetic DDIs based on target relationships
    # (drugs with targets in the same pathway)
    print("\nCreating additional synthetic DDIs...")

    # Group targets by pathway (if available)
    pathway_targets = defaultdict(set)
    if 'Pathway' in df_targets.columns:
        for idx, row in df_targets.iterrows():
            pathway = row.get('Pathway', '')
            if pd.isna(pathway) or pathway == '':
                continue

            target = row.get('UniProt ID', row.get('ID', ''))
            if target:
                pathway_targets[pathway].add(target)

        # Find drugs in same pathway
        pathway_drugs = defaultdict(set)
        for pathway, targets in pathway_targets.items():
            for target in targets:
                for drug in target_to_drugs.get(target, []):
                    pathway_drugs[pathway].add(drug)

        # Create DDIs for drugs in same pathway
        pathway_pairs = 0
        for pathway, pathway_drug_list in pathway_drugs.items():
            drug_list = list(pathway_drug_list)
            for i in range(len(drug_list)):
                for j in range(i + 1, len(drug_list)):
                    drug1 = drug_list[i]
                    drug2 = drug_list[j]

                    # Skip if already in ddi_pairs
                    if any((d['drug1_id'] == drug1 and d['drug2_id'] == drug2) or
                           (d['drug1_id'] == drug2 and d['drug2_id'] == drug1)
                           for d in ddi_pairs):
                        continue

                    ddi_pairs.append({
                        'drug1_id': drug1,
                        'drug2_id': drug2,
                        'interaction_type': 'target_related',
                        'description': f'Same pathway: {pathway}',
                        'severity': 'moderate',
                        'source': 'DrugBank_Pathways',
                        'confidence': 0.65
                    })
                    pathway_pairs += 1

        print(f"Created {pathway_pairs} additional pathway-based DDIs")

    # Create DataFrame
    df_ddi = pd.DataFrame(ddi_pairs)

    # Ensure required columns exist
    required_cols = ['drug1_id', 'drug2_id', 'interaction_type']
    for col in required_cols:
        if col not in df_ddi.columns:
            df_ddi[col] = ''

    # Save to CSV
    output_path = 'data/drugbank/ddi_data.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ddi.to_csv(output_path, index=False)

    print(f"\nSaved {len(df_ddi)} DDI pairs to: {output_path}")

    # Print statistics
    print(f"\nDDI Statistics:")
    print(f"  Total pairs: {len(df_ddi)}")
    print(f"  Unique drugs: {len(set(df_ddi['drug1_id'].tolist() + df_ddi['drug2_id'].tolist()))}")
    print(f"  Interaction types: {df_ddi['interaction_type'].unique().tolist()}")
    print(f"\nInteraction type distribution:")
    print(df_ddi['interaction_type'].value_counts())

    # Show sample
    print(f"\nSample DDI pairs:")
    print(df_ddi.head(10).to_string())

    return df_ddi


def create_binary_ddi_data():
    """Create binary DDI data (positive and negative samples)"""

    # First create positive samples
    df_positive = create_ddi_from_targets()

    if df_positive is None or len(df_positive) == 0:
        print("Error: Could not create positive samples")
        return

    # Load drug links to get all available drugs
    drug_links_path = 'data/drugbank/drug_links.csv'
    if os.path.exists(drug_links_path):
        df_drugs = pd.read_csv(drug_links_path)
        all_drugs = df_drugs['DrugBank ID'].dropna().tolist()
    else:
        # Use drugs from positive samples
        all_drugs = list(set(df_positive['drug1_id'].tolist() + df_positive['drug2_id'].tolist()))

    print(f"\nTotal drugs available: {len(all_drugs)}")

    # Create negative samples (non-interacting pairs)
    import random
    random.seed(42)

    negative_samples = []
    used_pairs = set()

    # Add existing pairs to used set
    for _, row in df_positive.iterrows():
        pair = tuple(sorted([row['drug1_id'], row['drug2_id']]))
        used_pairs.add(pair)

    # Number of negative samples = number of positive samples
    num_negative = min(len(df_positive), len(all_drugs) * 10)

    print(f"Creating {num_negative} negative samples...")

    attempts = 0
    while len(negative_samples) < num_negative and attempts < num_negative * 20:
        drug1 = random.choice(all_drugs)
        drug2 = random.choice(all_drugs)

        if drug1 == drug2:
            attempts += 1
            continue

        pair = tuple(sorted([drug1, drug2]))
        if pair in used_pairs:
            attempts += 1
            continue

        used_pairs.add(pair)

        negative_samples.append({
            'drug1_id': drug1,
            'drug2_id': drug2,
            'interaction_type': 'no_interaction',
            'description': 'No known interaction',
            'severity': 'none',
            'source': 'Synthetic',
            'confidence': 0.95
        })
        attempts += 1

    # Combine positive and negative
    all_ddi = pd.concat([df_positive, pd.DataFrame(negative_samples)], ignore_index=True)

    # Save
    output_path = 'data/drugbank/ddi_data_binary.csv'
    all_ddi.to_csv(output_path, index=False)

    print(f"\nSaved binary DDI data to: {output_path}")
    print(f"Positive samples: {len(df_positive)}")
    print(f"Negative samples: {len(negative_samples)}")
    print(f"Total: {len(all_ddi)}")

    return all_ddi


if __name__ == '__main__':
    # Option 1: Create DDI from targets
    create_ddi_from_targets()

    # Option 2: Create binary DDI data (uncomment to use)
    # create_binary_ddi_data()

    print("\nRun preprocessing now:")
    print("python scripts/preprocess_drugbank.py")