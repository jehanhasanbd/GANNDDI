# training/evaluator.py

"""
Model evaluator
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List

from utils.metrics import MetricsCalculator

"""
Model evaluator
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List

"""
Model evaluator
"""


class Evaluator:
    """Model evaluator"""

    def __init__(self, config: Dict):
        self.config = config
        self.metrics = MetricsCalculator()

    def evaluate(
            self,
            model: nn.Module,
            data_loader: DataLoader,
            device: str = 'cuda'
    ) -> Dict:
        """Evaluate model on dataset"""

        model.eval()
        model = model.to(device)

        all_preds = []
        all_labels = []
        all_scores = []
        total_loss = 0.0

        # Loss function
        num_classes = model.classifier[-1].out_features
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            pbar = tqdm(data_loader, desc='Evaluating')

            for batch in pbar:
                # Skip empty batches
                if batch['labels'].numel() == 0:
                    continue

                # Move batch to device
                batch = self._prepare_batch(batch, device)

                # Forward pass
                logits = model(
                    batch['node_features1'],
                    batch['edge_index1'],
                    batch['edge_features1'],
                    batch['node_features2'],
                    batch['edge_index2'],
                    batch['edge_features2'],
                    batch.get('batch1'),
                    batch.get('batch2')
                )

                loss = criterion(logits, batch['labels'])
                total_loss += loss.item()

                # Collect predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(batch['labels'].detach().cpu().numpy())
                all_scores.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / max(1, len(data_loader))

        if len(all_labels) > 0:
            metrics = self.metrics.compute_all_metrics(
                torch.tensor(all_labels),
                torch.tensor(all_preds),
                torch.tensor(all_scores) if all_scores else None
            )
        else:
            metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        metrics['loss'] = avg_loss

        return metrics

    def _prepare_batch(self, batch: Dict, device: str) -> Dict:
        """Prepare batch for evaluation"""
        prepared = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device)
            else:
                prepared[key] = value

        return prepared