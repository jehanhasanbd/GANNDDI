# training/loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class DDILoss(nn.Module):
    """Combined loss for DDI prediction"""

    def __init__(
            self,
            num_classes: int,
            focal_gamma: float = 2.0,
            label_smoothing: float = 0.1,
            contrastive_weight: float = 0.1
    ):
        super().__init__()

        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight

        # Cross-entropy loss with label smoothing
        self.ce_loss = LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=label_smoothing
        )

        # Focal loss for hard examples
        self.focal_loss = FocalLoss(
            gamma=focal_gamma
        )

    def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            features1: Optional[torch.Tensor] = None,
            features2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        logits: [batch_size, num_classes]
        labels: [batch_size]
        """

        # Main classification loss
        ce_loss = self.ce_loss(logits, labels)
        focal_loss = self.focal_loss(logits, labels)

        loss = ce_loss + focal_loss

        # Contrastive loss if features provided
        if features1 is not None and features2 is not None:
            contrastive_loss = self._contrastive_loss(
                features1, features2, labels
            )
            loss = loss + self.contrastive_weight * contrastive_loss

        return loss

    def _contrastive_loss(
            self,
            features1: torch.Tensor,
            features2: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss for drug pair representations"""

        # Normalize features
        features1 = F.normalize(features1, p=2, dim=-1)
        features2 = F.normalize(features2, p=2, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features1, features2.transpose(-2, -1))

        # Temperature scaling
        temperature = 0.07
        sim_matrix = sim_matrix / temperature

        # Positive pairs: same label
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.transpose(0, 1)).float()
        negative_mask = 1 - positive_mask

        # Contrastive loss
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        neg_sim = (exp_sim * negative_mask).sum(dim=1)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-6))

        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""

    def __init__(
            self,
            gamma: float = 2.0,
            alpha: Optional[torch.Tensor] = None,
            reduction: str = 'mean'
    ):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: [batch_size, num_classes]
        targets: [batch_size]
        """

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross-Entropy Loss"""

    def __init__(
            self,
            num_classes: int,
            smoothing: float = 0.1,
            reduction: str = 'mean'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: [batch_size, num_classes]
        targets: [batch_size]
        """

        # Convert to one-hot
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Apply label smoothing
        smoothed_labels = (1 - self.smoothing) * one_hot + self.smoothing / self.num_classes

        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(log_probs * smoothed_labels).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for drug representations"""

    def __init__(
            self,
            temperature: float = 0.07,
            reduction: str = 'mean'
    ):
        super().__init__()

        self.temperature = temperature
        self.reduction = reduction

    def forward(
            self,
            features1: torch.Tensor,
            features2: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        features1: [batch_size, hidden_dim]
        features2: [batch_size, hidden_dim]
        labels: [batch_size]
        """

        # Normalize features
        features1 = F.normalize(features1, p=2, dim=-1)
        features2 = F.normalize(features2, p=2, dim=-1)

        # Combine features
        features = torch.cat([features1, features2], dim=0)

        # Similarity matrix
        sim_matrix = torch.matmul(features, features.transpose(0, 1)) / self.temperature

        # Positive pairs: same label and same drug
        labels_combined = torch.cat([labels, labels], dim=0)
        positive_mask = (labels_combined.unsqueeze(1) == labels_combined.unsqueeze(0)).float()

        # Remove self-pairs
        identity_mask = torch.eye(len(features), device=features.device)
        positive_mask = positive_mask * (1 - identity_mask)

        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        neg_sim = (exp_sim * (1 - positive_mask - identity_mask)).sum(dim=1)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-6))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss