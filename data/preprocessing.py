# data/preprocessing.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Set
import json
from collections import defaultdict
import os
from rdkit.Chem import rdmolops


class Preprocessor:
    """Preprocess DrugBank data for DDI prediction"""

    def __init__(self, config):
        self.config = config
        self.drug_id_map = {}
        self.drug_smiles_map = {}
        self.ddi_map = {}
        self.label_encoder = LabelEncoder()
        self.drug_features = {}

    def load_drug_data(self, drug_links_path: str) -> pd.DataFrame:
        """Load drug links data"""
        if not os.path.exists(drug_links_path):
            print(f"Warning: {drug_links_path} not found")
            return pd.DataFrame()
        try:
            return pd.read_csv(drug_links_path)
        except Exception as e:
            print(f"Error loading {drug_links_path}: {e}")
            return pd.DataFrame()

    def load_drugbank_vocab(self, vocab_path: str) -> pd.DataFrame:
        """Load drugbank vocabulary"""
        return pd.read_csv(vocab_path, sep='\t')

    def load_structures(self, structures_path: str) -> pd.DataFrame:
        """
        Load drug structures from SDF file
        This parses the SDF file and extracts molecule information
        """
        print(f"Loading structures from SDF: {structures_path}")

        # Check if file exists
        if not os.path.exists(structures_path):
            print(f"Warning: {structures_path} not found")
            return pd.DataFrame()

        # Try to read as SDF
        try:
            suppl = Chem.SDMolSupplier(structures_path)

            structures = []

            for i, mol in enumerate(suppl):
                if mol is None:
                    continue

                # Get properties
                props = mol.GetPropsAsDict()

                # Extract SMILES
                smiles = Chem.MolToSmiles(mol)

                # Get DrugBank ID
                drugbank_id = props.get('DRUGBANK_ID', props.get('DATABASE_ID', f'DB{i:05d}'))

                # Get other properties
                structure = {
                    'DATABASE_ID': drugbank_id,
                    'DRUGBANK_ID': drugbank_id,
                    'SMILES': smiles,
                    'FORMULA': props.get('FORMULA', ''),
                    'MOLECULAR_WEIGHT': props.get('MOLECULAR_WEIGHT', 0.0),
                    'EXACT_MASS': props.get('EXACT_MASS', 0.0),
                    'INCHI_IDENTIFIER': props.get('INCHI_IDENTIFIER', ''),
                    'INCHI_KEY': props.get('INCHI_KEY', ''),
                    'GENERIC_NAME': props.get('GENERIC_NAME', ''),
                    'DRUG_GROUPS': props.get('DRUG_GROUPS', ''),
                    'SYNONYMS': props.get('SYNONYMS', ''),
                    'PRODUCTS': props.get('PRODUCTS', ''),
                    'INTERNATIONAL_BRANDS': props.get('INTERNATIONAL_BRANDS', ''),
                    'SECONDARY_ACCESSION_NUMBERS': props.get('SECONDARY_ACCESSION_NUMBERS', ''),
                    'JCHEM_ACCEPTOR_COUNT': props.get('JCHEM_ACCEPTOR_COUNT', 0),
                    'JCHEM_DONOR_COUNT': props.get('JCHEM_DONOR_COUNT', 0),
                    'JCHEM_NUMBER_OF_RINGS': props.get('JCHEM_NUMBER_OF_RINGS', 0),
                    'ALOGPS_LOGP': props.get('ALOGPS_LOGP', 0.0),
                    'JCHEM_POLAR_SURFACE_AREA': props.get('JCHEM_POLAR_SURFACE_AREA', 0.0),
                }

                # Compute additional properties if not present
                if 'JCHEM_ACCEPTOR_COUNT' not in props or props['JCHEM_ACCEPTOR_COUNT'] == '':
                    structure['JCHEM_ACCEPTOR_COUNT'] = Descriptors.NumHAcceptors(mol)

                if 'JCHEM_DONOR_COUNT' not in props or props['JCHEM_DONOR_COUNT'] == '':
                    structure['JCHEM_DONOR_COUNT'] = Descriptors.NumHDonors(mol)

                if 'JCHEM_NUMBER_OF_RINGS' not in props or props['JCHEM_NUMBER_OF_RINGS'] == '':
                    structure['JCHEM_NUMBER_OF_RINGS'] = Descriptors.RingCount(mol)

                if 'ALOGPS_LOGP' not in props or props['ALOGPS_LOGP'] == '':
                    structure['ALOGPS_LOGP'] = Descriptors.MolLogP(mol)

                if 'JCHEM_POLAR_SURFACE_AREA' not in props or props['JCHEM_POLAR_SURFACE_AREA'] == '':
                    structure['JCHEM_POLAR_SURFACE_AREA'] = Descriptors.TPSA(mol)

                structures.append(structure)

                # Print progress
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1} molecules...")

            print(f"Loaded {len(structures)} structures from SDF")

            # Convert to DataFrame
            df = pd.DataFrame(structures)

            # Remove duplicates
            df = df.drop_duplicates(subset=['DRUGBANK_ID'])

            return df

        except Exception as e:
            print(f"Error reading SDF file: {e}")
            print("Attempting alternative parsing method...")

            # Alternative: Try to parse SDF manually
            return self._parse_sdf_manually(structures_path)

    def _parse_sdf_manually(self, structures_path: str) -> pd.DataFrame:
        """Parse SDF file manually if SDMolSupplier fails"""

        structures = []

        try:
            with open(structures_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Split by molecule
            molecules = content.split('$$$$')

            for mol_text in molecules:
                if not mol_text.strip():
                    continue

                # Extract SMILES (often in the text or as a property)
                smiles = None
                drugbank_id = None
                props = {}

                lines = mol_text.strip().split('\n')

                for line in lines:
                    line = line.strip()

                    # Look for SMILES
                    if 'SMILES' in line and '=' in line:
                        try:
                            smiles = line.split('=')[1].strip()
                        except:
                            pass

                    # Look for DrugBank ID
                    if 'DRUGBANK_ID' in line and '=' in line:
                        try:
                            drugbank_id = line.split('=')[1].strip()
                        except:
                            pass

                    if 'DATABASE_ID' in line and '=' in line:
                        try:
                            if not drugbank_id:
                                drugbank_id = line.split('=')[1].strip()
                        except:
                            pass

                    # Look for other properties
                    if ':' in line or '=' in line:
                        parts = line.split('=' if '=' in line else ':')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            props[key] = value

                # If SMILES found, create molecule
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue

                    # Use drugbank_id or create one
                    if not drugbank_id:
                        drugbank_id = f"DB{len(structures):05d}"

                    # Create structure entry
                    structure = {
                        'DRUGBANK_ID': drugbank_id,
                        'DATABASE_ID': drugbank_id,
                        'SMILES': smiles,
                        'FORMULA': props.get('FORMULA', ''),
                        'MOLECULAR_WEIGHT': props.get('MOLECULAR_WEIGHT', 0.0),
                        'GENERIC_NAME': props.get('GENERIC_NAME', ''),
                        'DRUG_GROUPS': props.get('DRUG_GROUPS', ''),
                        'SYNONYMS': props.get('SYNONYMS', ''),
                        'JCHEM_ACCEPTOR_COUNT': Descriptors.NumHAcceptors(mol),
                        'JCHEM_DONOR_COUNT': Descriptors.NumHDonors(mol),
                        'JCHEM_NUMBER_OF_RINGS': Descriptors.RingCount(mol),
                        'ALOGPS_LOGP': Descriptors.MolLogP(mol),
                        'JCHEM_POLAR_SURFACE_AREA': Descriptors.TPSA(mol),
                    }

                    structures.append(structure)

            print(f"Manually parsed {len(structures)} structures")
            return pd.DataFrame(structures)

        except Exception as e:
            print(f"Manual parsing failed: {e}")
            return pd.DataFrame()

    def load_ddi_data(self, ddi_path: str) -> pd.DataFrame:
        """Load DDI data"""
        if os.path.exists(ddi_path):
            try:
                return pd.read_csv(ddi_path)
            except Exception as e:
                print(f"Error loading DDI data: {e}")
                return pd.DataFrame()
        else:
            print(f"Warning: DDI file not found at {ddi_path}")
            return pd.DataFrame()

    def preprocess_drugs(
            self,
            drug_links_df: pd.DataFrame,
            structures_df: pd.DataFrame
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """Preprocess drug information"""

        print("Preprocessing drugs...")

        # Create drug ID to SMILES mapping
        drug_smiles = {}
        drug_info = {}

        # First try to get from structures
        for _, row in structures_df.iterrows():
            drug_id = row.get('DRUGBANK_ID', row.get('DATABASE_ID', ''))
            if pd.isna(drug_id) or drug_id == '':
                continue

            smiles = row.get('SMILES', '')
            if pd.isna(smiles) or smiles == '':
                continue

            # Check if valid molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            drug_smiles[drug_id] = smiles

            # Store additional info
            drug_info[drug_id] = {
                'name': row.get('GENERIC_NAME', ''),
                'smiles': smiles,
                'formula': row.get('FORMULA', ''),
                'molecular_weight': row.get('MOLECULAR_WEIGHT', 0.0),
                'drug_groups': row.get('DRUG_GROUPS', ''),
                'synonyms': row.get('SYNONYMS', ''),
            }

        # Also get from drug_links for any missing drugs
        for _, row in drug_links_df.iterrows():
            drug_id = row.get('DrugBank ID', '')
            if pd.isna(drug_id) or drug_id == '':
                continue

            if drug_id not in drug_smiles:
                # Try to get from CAS number or other identifiers
                drug_name = row.get('Name', '')
                cas = row.get('CAS Number', '')

                # Try to get SMILES from PubChem if available
                pubchem_cid = row.get('PubChem Compound ID', '')

                # If we have a valid PubChem ID, we could fetch SMILES
                # For now, just skip
                pass

        print(f"Processed {len(drug_smiles)} drugs with SMILES")
        return drug_smiles, drug_info

    def build_ddi_pairs(
            self,
            drug_smiles: Dict[str, str],
            ddi_data: pd.DataFrame
    ) -> Tuple[List[Tuple[str, str, int]], Dict[str, int]]:
        """Build DDI pairs with labels"""

        print("Building DDI pairs...")

        ddi_pairs = []
        label_map = {}
        label_counter = defaultdict(int)

        if ddi_data.empty:
            print("Warning: No DDI data available. Creating synthetic DDIs...")
            return self._create_synthetic_ddi_pairs(drug_smiles)

        # Try different column name patterns
        drug1_col = None
        drug2_col = None
        interaction_col = None

        for col in ddi_data.columns:
            col_lower = col.lower()
            if 'drug1' in col_lower or 'drug_1' in col_lower or 'drug_a' in col_lower:
                drug1_col = col
            elif 'drug2' in col_lower or 'drug_2' in col_lower or 'drug_b' in col_lower:
                drug2_col = col
            elif 'interaction' in col_lower or 'type' in col_lower or 'label' in col_lower:
                interaction_col = col

        if drug1_col is None or drug2_col is None:
            print("Warning: Could not identify drug columns. Creating synthetic DDIs...")
            return self._create_synthetic_ddi_pairs(drug_smiles)

        for _, row in ddi_data.iterrows():
            drug1 = str(row.get(drug1_col, '')).strip()
            drug2 = str(row.get(drug2_col, '')).strip()

            if not drug1 or not drug2:
                continue

            if drug1 not in drug_smiles or drug2 not in drug_smiles:
                continue

            # Get interaction type
            if interaction_col:
                interaction = str(row.get(interaction_col, 'interaction')).strip()
                if pd.isna(interaction) or interaction == '':
                    interaction = 'interaction'
            else:
                interaction = 'interaction'

            # Map interaction to label
            if interaction not in label_map:
                label_map[interaction] = len(label_map)

            label = label_map[interaction]
            label_counter[interaction] += 1

            ddi_pairs.append((drug1, drug2, label))

        if len(ddi_pairs) == 0:
            print("Warning: No valid DDI pairs found. Creating synthetic DDIs...")
            return self._create_synthetic_ddi_pairs(drug_smiles)

        print(f"Created {len(ddi_pairs)} DDI pairs with {len(label_map)} interaction types")
        return ddi_pairs, label_map

    def _create_synthetic_ddi_pairs(
            self,
            drug_smiles: Dict[str, str]
    ) -> Tuple[List[Tuple[str, str, int]], Dict[str, int]]:
        """Create synthetic DDI pairs for testing"""

        print("Creating synthetic DDI pairs...")

        drugs = list(drug_smiles.keys())
        if len(drugs) < 2:
            print("Error: Not enough drugs to create synthetic pairs")
            return [], {}

        import random
        random.seed(42)

        ddi_pairs = []
        label_map = {'interaction': 0}
        used_pairs = set()

        # Create positive pairs (50% of total)
        num_pairs = min(500, len(drugs) * 5)
        num_positive = num_pairs // 2

        for _ in range(num_positive):
            attempts = 0
            while attempts < 100:
                drug1 = random.choice(drugs)
                drug2 = random.choice(drugs)

                if drug1 == drug2:
                    attempts += 1
                    continue

                pair = tuple(sorted([drug1, drug2]))
                if pair in used_pairs:
                    attempts += 1
                    continue

                used_pairs.add(pair)
                ddi_pairs.append((drug1, drug2, 0))
                break

        # Create negative pairs (50% of total)
        num_negative = min(num_positive, len(drugs) * 5)

        for _ in range(num_negative):
            attempts = 0
            while attempts < 100:
                drug1 = random.choice(drugs)
                drug2 = random.choice(drugs)

                if drug1 == drug2:
                    attempts += 1
                    continue

                pair = tuple(sorted([drug1, drug2]))
                if pair in used_pairs:
                    attempts += 1
                    continue

                used_pairs.add(pair)
                ddi_pairs.append((drug1, drug2, 1))  # 1 for no interaction
                break

        print(f"Created {len(ddi_pairs)} synthetic DDI pairs")
        return ddi_pairs, label_map

    def create_dataset(
            self,
            ddi_pairs: List[Tuple[str, str, int]],
            drug_smiles: Dict[str, str],
            drug_info: Dict[str, Dict]
    ) -> List[Dict]:
        """Create dataset entries for model training"""

        dataset = []

        for drug1, drug2, label in ddi_pairs:
            entry = {
                'drug1_id': drug1,
                'drug2_id': drug2,
                'drug1_smiles': drug_smiles.get(drug1, ''),
                'drug2_smiles': drug_smiles.get(drug2, ''),
                'drug1_info': drug_info.get(drug1, {}),
                'drug2_info': drug_info.get(drug2, {}),
                'label': label
            }
            dataset.append(entry)

        return dataset

    def filter_dataset(
            self,
            dataset: List[Dict],
            max_atoms: int = 150,
            max_mol_weight: float = 2000.0
    ) -> List[Dict]:
        """Filter dataset based on molecular properties"""

        filtered = []

        for entry in dataset:
            drug1_smiles = entry.get('drug1_smiles', '')
            drug2_smiles = entry.get('drug2_smiles', '')

            if not drug1_smiles or not drug2_smiles:
                continue

            # Check molecular properties
            mol1 = Chem.MolFromSmiles(drug1_smiles)
            mol2 = Chem.MolFromSmiles(drug2_smiles)

            if mol1 is None or mol2 is None:
                continue

            if mol1.GetNumAtoms() > max_atoms or mol2.GetNumAtoms() > max_atoms:
                continue

            mol_weight1 = Descriptors.MolWt(mol1)
            mol_weight2 = Descriptors.MolWt(mol2)

            if mol_weight1 > max_mol_weight or mol_weight2 > max_mol_weight:
                continue

            filtered.append(entry)

        print(f"Filtered {len(dataset)} -> {len(filtered)} entries")
        return filtered

    def split_dataset(
            self,
            dataset: List[Dict],
            test_size: float = 0.2,
            val_size: float = 0.1,
            random_state: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train, validation, and test sets"""

        if len(dataset) == 0:
            return [], [], []

        # Get labels for stratification
        labels = [entry['label'] for entry in dataset]

        # First split: train+val and test
        train_val, test = train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        train_labels = [entry['label'] for entry in train_val]

        try:
            train, val = train_test_split(
                train_val,
                test_size=val_ratio,
                random_state=random_state,
                stratify=train_labels
            )
        except ValueError:
            # If stratification fails (too few samples), use random split
            train, val = train_test_split(
                train_val,
                test_size=val_ratio,
                random_state=random_state
            )

        print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
        return train, val, test

    def create_drug_feature_matrices(
            self,
            drug_smiles: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """Create feature matrices for drugs"""

        drug_features = {}

        for drug_id, smiles in drug_smiles.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Create Morgan fingerprints
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            features = np.array(fp)

            drug_features[drug_id] = features

        return drug_features

    def save_preprocessed_data(
            self,
            output_dir: str,
            train_data: List[Dict],
            val_data: List[Dict],
            test_data: List[Dict],
            drug_smiles: Dict[str, str],
            drug_info: Dict[str, Dict],
            label_map: Dict[str, int]
    ):
        """Save preprocessed data to disk"""

        os.makedirs(output_dir, exist_ok=True)

        # Save datasets
        pd.DataFrame(train_data).to_csv(f"{output_dir}/train.csv", index=False)
        pd.DataFrame(val_data).to_csv(f"{output_dir}/val.csv", index=False)
        pd.DataFrame(test_data).to_csv(f"{output_dir}/test.csv", index=False)

        # Save drug info
        with open(f"{output_dir}/drug_smiles.json", 'w') as f:
            json.dump(drug_smiles, f)

        with open(f"{output_dir}/drug_info.json", 'w') as f:
            json.dump(drug_info, f)

        with open(f"{output_dir}/label_map.json", 'w') as f:
            json.dump(label_map, f)

        print(f"Preprocessed data saved to {output_dir}")

    def load_preprocessed_data(self, data_dir: str) -> Dict:
        """Load preprocessed data from disk"""

        train_data = pd.read_csv(f"{data_dir}/train.csv").to_dict('records')
        val_data = pd.read_csv(f"{data_dir}/val.csv").to_dict('records')
        test_data = pd.read_csv(f"{data_dir}/test.csv").to_dict('records')

        with open(f"{data_dir}/drug_smiles.json", 'r') as f:
            drug_smiles = json.load(f)

        with open(f"{data_dir}/drug_info.json", 'r') as f:
            drug_info = json.load(f)

        with open(f"{data_dir}/label_map.json", 'r') as f:
            label_map = json.load(f)

        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'drug_smiles': drug_smiles,
            'drug_info': drug_info,
            'label_map': label_map
        }