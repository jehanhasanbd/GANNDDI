# experiments/hyperparameter_tuning.py

import torch
import itertools
import json
import os
from typing import Dict, List, Any
from sklearn.model_selection import ParameterGrid
import optuna


from config.model_config import get_default_config, ModelConfig
from data.drugbank_loader import DrugBankLoader
from data.dataset import create_dataloaders
from models.ddi_predictor import DDIPredictor
from training.trainer import Trainer


class HyperparameterTuner:
    """Hyperparameter tuning for GANNDDI"""

    def __init__(
            self,
            config: Dict,
            n_trials: int = 20,
            study_name: str = 'gannddi_tuning'
    ):
        self.config = config
        self.n_trials = n_trials
        self.study_name = study_name

    def create_study(self) -> optuna.Study:
        """Create Optuna study"""
        return optuna.create_study(
            study_name=self.study_name,
            direction='maximize'
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""

        # Suggest hyperparameters
        hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=32)
        num_heads = trial.suggest_int('num_heads', 4, 16, step=2)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        num_patterns = trial.suggest_int('num_patterns', 8, 32, step=4)
        dropout = trial.suggest_float('dropout', 0.05, 0.5, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        # Update config
        self.config['model']['gaan']['hidden_dim'] = hidden_dim
        self.config['model']['gaan']['num_heads'] = num_heads
        self.config['model']['gaan']['num_layers'] = num_layers
        self.config['model']['sie_encoder']['num_patterns'] = num_patterns
        self.config['model']['ddi_predictor']['dropout'] = dropout
        self.config['training']['learning_rate'] = learning_rate
        self.config['training']['weight_decay'] = weight_decay
        self.config['data']['batch_size'] = batch_size

        # Load data
        loader = DrugBankLoader(self.config)
        data = loader.load_all_data()

        train_data = data['train_data']
        val_data = data['val_data']
        test_data = data['test_data']
        drug_smiles = data['drug_smiles']
        label_map = data['label_map']

        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            train_data, val_data, test_data,
            drug_smiles, batch_size, 2, 150
        )

        # Initialize model
        num_classes = len(label_map)
        model = DDIPredictor(
            node_feat_dim=74,
            edge_feat_dim=10,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_patterns=num_patterns,
            dropout=dropout,
            num_classes=num_classes
        )

        # Train
        trainer = Trainer(
            model=model,
            config=self.config,
            use_wandb=False
        )

        # Quick training for tuning
        trainer.epochs = min(5, trainer.epochs)
        trainer.early_stopping_patience = 3

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            save_dir='tuning_checkpoints/'
        )

        # Return best validation score
        best_val_score = max([h.get('f1', h.get('accuracy', 0.0)) for h in trainer.val_history])

        return best_val_score

    def run_optuna_tuning(self) -> Dict:
        """Run Optuna hyperparameter tuning"""

        study = self.create_study()
        study.optimize(self.objective, n_trials=self.n_trials)

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        # Save results
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }

        with open('tuning_results.json', 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_value': best_value,
                'all_trials': [
                    {
                        'params': t.params,
                        'value': t.value
                    }
                    for t in study.trials
                ]
            }, f, indent=2)

        return results

    def run_grid_search(self, param_grid: Dict) -> Dict:
        """Run grid search hyperparameter tuning"""

        best_score = 0.0
        best_params = None
        results = []

        # Create parameter combinations
        param_combinations = list(ParameterGrid(param_grid))

        for params in param_combinations:
            print(f"Testing parameters: {params}")

            # Update config
            for key, value in params.items():
                keys = key.split('__')
                current = self.config
                for k in keys[:-1]:
                    if k in current:
                        current = current[k]
                current[keys[-1]] = value

            # Train and evaluate
            # (Similar to objective function)
            # ...

            # Store results
            results.append({
                'params': params,
                'score': 0.0  # Replace with actual score
            })

        return {
            'best_params': best_params,
            'best_score': best_score,
            'results': results
        }