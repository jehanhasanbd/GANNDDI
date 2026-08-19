# utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Optional


class MetricsCalculator:
    """Calculate various metrics for model evaluation"""

    @staticmethod
    def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

    @staticmethod
    def precision(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = 'macro') -> float:
        return precision_score(
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
            average=average,
            zero_division=0
        )

    @staticmethod
    def recall(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = 'macro') -> float:
        return recall_score(
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
            average=average,
            zero_division=0
        )

    @staticmethod
    def f1(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = 'macro') -> float:
        return f1_score(
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
            average=average,
            zero_division=0
        )

    @staticmethod
    def auroc(y_true: torch.Tensor, y_score: torch.Tensor, multi_class: str = 'ovr') -> float:
        if y_score.shape[1] > 1:
            # Multi-class
            return roc_auc_score(
                y_true.cpu().numpy(),
                y_score.cpu().numpy(),
                multi_class=multi_class,
                average='macro'
            )
        else:
            # Binary
            return roc_auc_score(
                y_true.cpu().numpy(),
                y_score.cpu().numpy()
            )

    @staticmethod
    def auprc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
        if y_score.shape[1] > 1:
            # Multi-class - average over classes
            scores = []
            for i in range(y_score.shape[1]):
                y_true_binary = (y_true == i).cpu().numpy().astype(int)
                scores.append(average_precision_score(
                    y_true_binary,
                    y_score[:, i].cpu().numpy()
                ))
            return np.mean(scores)
        else:
            # Binary
            return average_precision_score(
                y_true.cpu().numpy(),
                y_score.cpu().numpy()
            )

    @staticmethod
    def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
        return confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())

    @staticmethod
    def classification_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
        return classification_report(
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
            zero_division=0
        )

    @classmethod
    def compute_all_metrics(
            cls,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            y_score: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {
            'accuracy': cls.accuracy(y_true, y_pred),
            'precision': cls.precision(y_true, y_pred),
            'recall': cls.recall(y_true, y_pred),
            'f1': cls.f1(y_true, y_pred),
        }

        if y_score is not None:
            try:
                metrics['auroc'] = cls.auroc(y_true, y_score)
                metrics['auprc'] = cls.auprc(y_true, y_score)
            except Exception:
                pass

        return metrics