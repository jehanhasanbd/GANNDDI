# utils/chem_utils.py

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from typing import Optional, Tuple


class ChemistryUtils:
    """Utility functions for chemistry operations"""

    @staticmethod
    def validate_smiles(smiles: str) -> bool:
        """Validate SMILES string"""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    @staticmethod
    def get_molecular_weight(smiles: str) -> Optional[float]:
        """Get molecular weight from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Descriptors.MolWt(mol)

    @staticmethod
    def get_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[bytearray]:
        """Get Morgan fingerprint from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    @staticmethod
    def get_atom_count(smiles: str) -> Optional[int]:
        """Get number of atoms from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol.GetNumAtoms()

    @staticmethod
    def get_ring_count(smiles: str) -> Optional[int]:
        """Get number of rings from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Descriptors.RingCount(mol)