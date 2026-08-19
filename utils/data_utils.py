# utils/data_utils.py

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json


class DataUtils:
    """Utility functions for data processing"""

    @staticmethod
    def to_tensor(data: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert numpy array to torch tensor"""
        return torch.tensor(data, dtype=dtype)

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array"""
        return tensor.detach().cpu().numpy()

    @staticmethod
    def save_json(data: Dict, filepath: str):
        """Save data as JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load JSON data"""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_pickle(data: object, filepath: str):
        """Save data as pickle"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_pickle(filepath: str) -> object:
        """Load pickle data"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)