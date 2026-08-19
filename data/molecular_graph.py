# data/molecular_graph.py

"""
Molecular graph representation for DDI prediction
"""

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MolecularGraph:
    """Molecular graph representation for DDI prediction"""

    # Node features
    node_features: torch.Tensor  # [num_nodes, feat_dim]
    # Edge features
    edge_index: torch.Tensor  # [2, num_edges]
    edge_features: torch.Tensor  # [num_edges, edge_feat_dim]
    # Graph-level features
    graph_features: torch.Tensor  # [feat_dim]
    # Atom and bond information
    atom_symbols: List[str]
    bond_types: List[str]
    # SMILES string
    smiles: str
    # Drug identifier
    drug_id: str
    # Additional properties
    molecular_weight: float
    num_atoms: int
    num_bonds: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'node_features': self.node_features,
            'edge_index': self.edge_index,
            'edge_features': self.edge_features,
            'graph_features': self.graph_features,
            'atom_symbols': self.atom_symbols,
            'bond_types': self.bond_types,
            'smiles': self.smiles,
            'drug_id': self.drug_id,
            'molecular_weight': self.molecular_weight,
            'num_atoms': self.num_atoms,
            'num_bonds': self.num_bonds
        }

    @classmethod
    def from_smiles(
            cls,
            smiles: str,
            drug_id: str = "",
            max_atoms: int = 150,
            atom_feat_dim: int = 74,
            bond_feat_dim: int = 10
    ) -> Optional['MolecularGraph']:
        """Create MolecularGraph from SMILES string"""

        if not smiles or smiles == '':
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            # Sanitize molecule
            Chem.SanitizeMol(mol)
        except:
            # If sanitization fails, try without sanitization
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

        # Check atom count
        num_atoms_actual = mol.GetNumAtoms()
        if num_atoms_actual > max_atoms:
            return None

        # Get molecular weight
        try:
            mol_weight = Descriptors.MolWt(mol)
        except:
            mol_weight = 0.0

        # Extract node features
        node_features = []
        atom_symbols = []

        for atom in mol.GetAtoms():
            features = []

            # 1. Atom type (one-hot of common elements)
            atomic_num = atom.GetAtomicNum()
            atom_type = atom.GetSymbol()
            atom_symbols.append(atom_type)

            # Common elements: H, C, N, O, F, P, S, Cl, Br, I, others
            common_elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
            for elem in common_elements:
                features.append(1.0 if atomic_num == elem else 0.0)
            # Other elements
            features.append(1.0 if atomic_num not in common_elements else 0.0)

            # 2. Degree (0-10)
            degree = atom.GetDegree()
            for d in range(11):
                features.append(1.0 if degree == d else 0.0)

            # 3. Valence (0-10)
            try:
                valence = atom.GetTotalValence()
            except:
                valence = 0
            for v in range(11):
                features.append(1.0 if valence == v else 0.0)

            # 4. Implicit valence (0-10)
            try:
                implicit_valence = atom.GetImplicitValence()
            except:
                implicit_valence = 0
            for v in range(11):
                features.append(1.0 if implicit_valence == v else 0.0)

            # 5. Formal charge (-4 to +4)
            formal_charge = atom.GetFormalCharge()
            for c in range(-4, 5):
                features.append(1.0 if formal_charge == c else 0.0)

            # 6. Hybridization
            try:
                hyb = atom.GetHybridization()
                hyb_map = {
                    Chem.rdchem.HybridizationType.SP: 0,
                    Chem.rdchem.HybridizationType.SP2: 1,
                    Chem.rdchem.HybridizationType.SP3: 2,
                    Chem.rdchem.HybridizationType.SP3D: 3,
                    Chem.rdchem.HybridizationType.SP3D2: 4
                }
                hyb_onehot = [0] * 5
                hyb_idx = hyb_map.get(hyb, -1)
                if hyb_idx >= 0:
                    hyb_onehot[hyb_idx] = 1
                features.extend(hyb_onehot)
            except:
                features.extend([0] * 5)

            # 7. Aromaticity
            features.append(1.0 if atom.GetIsAromatic() else 0.0)

            # 8. Ring membership
            try:
                features.append(1.0 if atom.IsInRing() else 0.0)
            except:
                features.append(0.0)

            # 9. Chirality
            try:
                chirality = 1.0 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0
                features.append(chirality)
            except:
                features.append(0.0)

            node_features.append(features)

        # Get actual feature dimension
        actual_feat_dim = len(node_features[0]) if node_features else 74

        # Pad node features to max_atoms
        node_features = np.array(node_features, dtype=np.float32)
        if node_features.shape[0] < max_atoms:
            padding = np.zeros((max_atoms - node_features.shape[0], actual_feat_dim), dtype=np.float32)
            node_features = np.concatenate([node_features, padding], axis=0)

        # Extract edge features
        edge_index = []
        edge_features = []
        bond_types = []

        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # Add both directions for undirected graph
            edge_index.extend([[begin_idx, end_idx], [end_idx, begin_idx]])

            # Bond features
            bond_type = bond.GetBondType()
            # Bond features - always use 10 dimensions
            bond_type_onehot = [0] * 10  # Use 10 dimensions
            bond_type_map = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3
            }
            bond_type_idx = bond_type_map.get(bond_type, -1)
            if bond_type_idx >= 0:
                bond_type_onehot[bond_type_idx] = 1

            # Add both directions
            edge_features.extend([bond_type_onehot, bond_type_onehot])
            bond_types.extend([str(bond_type), str(bond_type)])

        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.tensor([[], []],
                                                                                                    dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros(
            (0, bond_feat_dim))
        node_features = torch.tensor(node_features, dtype=torch.float32)

        # Graph-level features
        graph_features = cls._get_graph_features(mol)

        return cls(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            graph_features=graph_features,
            atom_symbols=atom_symbols,
            bond_types=bond_types,
            smiles=smiles,
            drug_id=drug_id,
            molecular_weight=mol_weight,
            num_atoms=len(atom_symbols),
            num_bonds=len(bond_types) // 2
        )

    @staticmethod
    def _get_graph_features(mol: Chem.Mol) -> torch.Tensor:
        """Extract graph-level features"""
        features = []

        try:
            # Molecular weight (log transformed)
            mol_weight = Descriptors.MolWt(mol)
            features.append(np.log(mol_weight + 1))
        except:
            features.append(0.0)

        try:
            # Number of rings
            features.append(float(Descriptors.RingCount(mol)))
        except:
            features.append(0.0)

        try:
            # Number of rotatable bonds
            features.append(float(Descriptors.NumRotatableBonds(mol)))
        except:
            features.append(0.0)

        try:
            # Number of hydrogen bond donors
            features.append(float(Descriptors.NumHDonors(mol)))
        except:
            features.append(0.0)

        try:
            # Number of hydrogen bond acceptors
            features.append(float(Descriptors.NumHAcceptors(mol)))
        except:
            features.append(0.0)

        try:
            # LogP
            features.append(float(Descriptors.MolLogP(mol)))
        except:
            features.append(0.0)

        try:
            # Polar surface area
            features.append(float(Descriptors.TPSA(mol)))
        except:
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float32)


def graph_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching molecular graphs"""

    # Filter out None graphs
    valid_batch = []
    for item in batch:
        graph1 = item.get('graph1')
        graph2 = item.get('graph2')
        if graph1 is not None and graph2 is not None:
            valid_batch.append(item)

    if len(valid_batch) == 0:
        # Return empty batch
        return {
            'node_features1': torch.zeros(0, 1, 74),
            'edge_index1': torch.zeros(0, 2, 1),
            'edge_features1': torch.zeros(0, 1, 10),
            'node_features2': torch.zeros(0, 1, 74),
            'edge_index2': torch.zeros(0, 2, 1),
            'edge_features2': torch.zeros(0, 1, 10),
            'labels': torch.zeros(0, dtype=torch.long),
            'num_nodes1': torch.zeros(0),
            'num_nodes2': torch.zeros(0),
        }

    # Separate graphs and labels
    graphs1 = [item['graph1'] for item in valid_batch]
    graphs2 = [item['graph2'] for item in valid_batch]
    labels = torch.tensor([item['label'] for item in valid_batch], dtype=torch.long)

    # Get dimensions - FIXED: handle varying edge feature dimensions
    max_nodes1 = max([g.node_features.size(0) for g in graphs1])
    max_nodes2 = max([g.node_features.size(0) for g in graphs2])
    node_feat_dim1 = graphs1[0].node_features.size(1)
    node_feat_dim2 = graphs2[0].node_features.size(1)

    # Get edge feature dimensions - handle varying sizes
    edge_feat_dims1 = [g.edge_features.size(1) if g.edge_features.numel() > 0 else 0 for g in graphs1]
    edge_feat_dims2 = [g.edge_features.size(1) if g.edge_features.numel() > 0 else 0 for g in graphs2]

    # Use the maximum edge feature dimension
    edge_feat_dim1 = max(edge_feat_dims1) if edge_feat_dims1 else 10
    edge_feat_dim2 = max(edge_feat_dims2) if edge_feat_dims2 else 10

    # Pad node features
    padded_node_features1 = []
    padded_node_features2 = []
    for g1, g2 in zip(graphs1, graphs2):
        # Pad graph1
        if g1.node_features.size(0) < max_nodes1:
            padding1 = torch.zeros((max_nodes1 - g1.node_features.size(0), node_feat_dim1), dtype=torch.float32)
            padded1 = torch.cat([g1.node_features, padding1], dim=0)
        else:
            padded1 = g1.node_features
        padded_node_features1.append(padded1)

        # Pad graph2
        if g2.node_features.size(0) < max_nodes2:
            padding2 = torch.zeros((max_nodes2 - g2.node_features.size(0), node_feat_dim2), dtype=torch.float32)
            padded2 = torch.cat([g2.node_features, padding2], dim=0)
        else:
            padded2 = g2.node_features
        padded_node_features2.append(padded2)

    node_features1 = torch.stack(padded_node_features1, dim=0)
    node_features2 = torch.stack(padded_node_features2, dim=0)

    # Pad edge features - FIXED: ensure consistent edge feature dimensions
    max_edges1 = max([g.edge_index.size(1) for g in graphs1]) if graphs1 else 0
    max_edges2 = max([g.edge_index.size(1) for g in graphs2]) if graphs2 else 0

    padded_edge_index1 = []
    padded_edge_features1 = []
    padded_edge_index2 = []
    padded_edge_features2 = []

    for g1, g2 in zip(graphs1, graphs2):
        # Pad graph1 edges
        num_edges1 = g1.edge_index.size(1)
        if num_edges1 < max_edges1:
            padding_idx1 = torch.full((2, max_edges1 - num_edges1), 0, dtype=torch.long)
            padded_edge_idx1 = torch.cat([g1.edge_index, padding_idx1], dim=1)

            # FIXED: Create padding with correct edge feature dimension
            if g1.edge_features.numel() > 0:
                current_edge_feat_dim = g1.edge_features.size(1)
            else:
                current_edge_feat_dim = edge_feat_dim1

            padding_feat1 = torch.zeros((max_edges1 - num_edges1, current_edge_feat_dim), dtype=torch.float32)
            padded_edge_feat1 = torch.cat([g1.edge_features, padding_feat1], dim=0)
        else:
            padded_edge_idx1 = g1.edge_index
            padded_edge_feat1 = g1.edge_features

        # Pad graph2 edges
        num_edges2 = g2.edge_index.size(1)
        if num_edges2 < max_edges2:
            padding_idx2 = torch.full((2, max_edges2 - num_edges2), 0, dtype=torch.long)
            padded_edge_idx2 = torch.cat([g2.edge_index, padding_idx2], dim=1)

            # FIXED: Create padding with correct edge feature dimension
            if g2.edge_features.numel() > 0:
                current_edge_feat_dim2 = g2.edge_features.size(1)
            else:
                current_edge_feat_dim2 = edge_feat_dim2

            padding_feat2 = torch.zeros((max_edges2 - num_edges2, current_edge_feat_dim2), dtype=torch.float32)
            padded_edge_feat2 = torch.cat([g2.edge_features, padding_feat2], dim=0)
        else:
            padded_edge_idx2 = g2.edge_index
            padded_edge_feat2 = g2.edge_features

        padded_edge_index1.append(padded_edge_idx1)
        padded_edge_features1.append(padded_edge_feat1)
        padded_edge_index2.append(padded_edge_idx2)
        padded_edge_features2.append(padded_edge_feat2)

    # Stack edges - FIXED: pad to the same feature dimension if needed
    # Get max edge feature dimension across all graphs in the batch
    max_edge_feat_dim1 = max([feat.size(1) if feat.numel() > 0 else 0 for feat in
                              padded_edge_features1]) if padded_edge_features1 else edge_feat_dim1
    max_edge_feat_dim2 = max([feat.size(1) if feat.numel() > 0 else 0 for feat in
                              padded_edge_features2]) if padded_edge_features2 else edge_feat_dim2

    # Pad edge features to the same dimension
    padded_edge_features1_final = []
    padded_edge_features2_final = []

    for feat in padded_edge_features1:
        if feat.numel() > 0 and feat.size(1) < max_edge_feat_dim1:
            padding = torch.zeros((feat.size(0), max_edge_feat_dim1 - feat.size(1)), dtype=torch.float32)
            feat = torch.cat([feat, padding], dim=1)
        elif feat.numel() == 0:
            feat = torch.zeros((0, max_edge_feat_dim1), dtype=torch.float32)
        padded_edge_features1_final.append(feat)

    for feat in padded_edge_features2:
        if feat.numel() > 0 and feat.size(1) < max_edge_feat_dim2:
            padding = torch.zeros((feat.size(0), max_edge_feat_dim2 - feat.size(1)), dtype=torch.float32)
            feat = torch.cat([feat, padding], dim=1)
        elif feat.numel() == 0:
            feat = torch.zeros((0, max_edge_feat_dim2), dtype=torch.float32)
        padded_edge_features2_final.append(feat)

    edge_index1 = torch.stack(padded_edge_index1, dim=0)
    edge_features1 = torch.stack(padded_edge_features1_final, dim=0)
    edge_index2 = torch.stack(padded_edge_index2, dim=0)
    edge_features2 = torch.stack(padded_edge_features2_final, dim=0)

    # Stack graph features
    graph_features1 = torch.stack([g.graph_features for g in graphs1], dim=0)
    graph_features2 = torch.stack([g.graph_features for g in graphs2], dim=0)

    return {
        'node_features1': node_features1,  # [batch_size, max_nodes1, feat_dim]
        'edge_index1': edge_index1,  # [batch_size, 2, max_edges1]
        'edge_features1': edge_features1,  # [batch_size, max_edges1, edge_feat_dim]
        'node_features2': node_features2,  # [batch_size, max_nodes2, feat_dim]
        'edge_index2': edge_index2,  # [batch_size, 2, max_edges2]
        'edge_features2': edge_features2,  # [batch_size, max_edges2, edge_feat_dim]
        'graph_features1': graph_features1,  # [batch_size, graph_feat_dim]
        'graph_features2': graph_features2,  # [batch_size, graph_feat_dim]
        'labels': labels,  # [batch_size]
        'num_nodes1': torch.tensor([g.num_atoms for g in graphs1], dtype=torch.long),
        'num_nodes2': torch.tensor([g.num_atoms for g in graphs2], dtype=torch.long),
        'drug_ids1': [g.drug_id for g in graphs1],
        'drug_ids2': [g.drug_id for g in graphs2],
        'smiles1': [g.smiles for g in graphs1],
        'smiles2': [g.smiles for g in graphs2]
    }