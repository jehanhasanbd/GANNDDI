# training/trainer.py

"""
Training engine for GANNDDI
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Dict, Optional, List

# Use absolute imports
from training.evaluator import Evaluator
from training.loss_functions import DDILoss
from utils.metrics import MetricsCalculator

"""
Training engine for GANNDDI
"""

class Trainer:
    """Training engine for GANNDDI"""

    def __init__(
            self,
            model: nn.Module,
            config: Dict,
            device: Optional[str] = None,
            use_wandb: bool = False
    ):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb

        # Move model to device
        self.model = self.model.to(self.device)

        # Training config with proper type conversion
        train_config = config.get('training', {})

        # Ensure all numeric values are properly typed
        self.epochs = int(train_config.get('epochs', 100))
        self.learning_rate = float(train_config.get('learning_rate', 1e-4))
        self.weight_decay = float(train_config.get('weight_decay', 1e-5))
        self.warmup_steps = int(train_config.get('warmup_steps', 1000))
        self.gradient_clip = float(train_config.get('gradient_clip', 1.0))
        self.early_stopping_patience = int(train_config.get('early_stopping_patience', 10))

        # Print training config
        print(f"Training config: lr={self.learning_rate}, weight_decay={self.weight_decay}, epochs={self.epochs}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.warmup_steps,
            T_mult=2
        )

        # Loss function
        num_classes = self.model.classifier[-1].out_features
        self.criterion = DDILoss(num_classes=num_classes)

        # Metrics
        self.metrics = MetricsCalculator()
        self.evaluator = Evaluator(config)

        # Training state
        self.best_val_score = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_scores = []

        pbar = tqdm(train_loader, desc='Training')

        for batch in pbar:
            # Skip empty batches
            if batch['labels'].numel() == 0:
                continue

            # Move batch to device
            batch = self._prepare_batch(batch)

            # Forward pass (no mixed precision to avoid dtype issues)
            logits = self.model(
                batch['node_features1'],
                batch['edge_index1'],
                batch['edge_features1'],
                batch['node_features2'],
                batch['edge_index2'],
                batch['edge_features2'],
                batch.get('batch1'),
                batch.get('batch2')
            )

            loss = self.criterion(logits, batch['labels'])

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            # Update scheduler
            self.scheduler.step()

            # Collect metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(batch['labels'].detach().cpu().numpy())
            all_scores.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Compute metrics
        avg_loss = total_loss / max(1, len(train_loader))

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

    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate the model"""
        return self.evaluator.evaluate(self.model, val_loader, self.device)

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: Optional[DataLoader] = None,
            save_dir: str = 'checkpoints/'
    ) -> Dict:
        """Full training loop"""

        os.makedirs(save_dir, exist_ok=True)
        best_val_metrics = None

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Save history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)

            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Log to wandb
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'],
                        'train_f1': train_metrics['f1'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'val_f1': val_metrics['f1'],
                        'val_auroc': val_metrics.get('auroc', 0.0),
                        'val_auprc': val_metrics.get('auprc', 0.0),
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })
                except:
                    pass

            # Early stopping
            val_score = val_metrics.get('f1', val_metrics.get('accuracy', 0.0))

            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_model_state = self.model.state_dict()
                self.patience_counter = 0
                best_val_metrics = val_metrics

                # Save best model
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'config': self.config,
                        'num_classes': self.model.classifier[-1].out_features  # Add this line
                    },
                    os.path.join(save_dir, 'best_model.pt')
                )
                print(f"Best model saved with F1: {val_score:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Test if test_loader is provided
        test_metrics = None
        if test_loader is not None:
            test_metrics = self.validate(test_loader)
            print(f"\nTest Results: Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

            # Save test results
            with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
                json.dump(test_metrics, f, indent=2)

        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_val': best_val_metrics,
            'test': test_metrics
        }

        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        return history

    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare batch for training"""
        prepared = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value

        return prepared