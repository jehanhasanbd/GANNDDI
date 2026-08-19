# config/model_config.py

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os


@dataclass
class GANConfig:
    num_layers: int = 3
    num_heads: int = 8
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.2
    gate_dim: int = 16


@dataclass
class GATEEncoderConfig:
    num_heads: int = 8
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class SIEEncoderConfig:
    num_patterns: int = 16
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class DDIPredictorConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2


@dataclass
class DataConfig:
    drugbank_path: str = "data/drugbank/"
    batch_size: int = 32
    num_workers: int = 4
    max_atoms: int = 150
    max_mol_weight: float = 2000.0


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10


@dataclass
class EvaluationConfig:
    metrics: list = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auroc", "auprc"])
    test_split: float = 0.2
    val_split: float = 0.1


@dataclass
class ModelConfig:
    model: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(
            model=config_dict.get('model', {}),
            data=config_dict.get('data', {}),
            training=config_dict.get('training', {}),
            evaluation=config_dict.get('evaluation', {})
        )

    def get_gaan_config(self) -> GANConfig:
        gaan_params = self.model.get('gaan', {})
        return GANConfig(
            num_layers=int(gaan_params.get('num_layers', 3)),
            num_heads=int(gaan_params.get('num_heads', 8)),
            hidden_dim=int(gaan_params.get('hidden_dim', 256)),
            output_dim=int(gaan_params.get('output_dim', 128)),
            dropout=float(gaan_params.get('dropout', 0.2)),
            gate_dim=int(gaan_params.get('gate_dim', 16))
        )

    def get_gate_encoder_config(self) -> GATEEncoderConfig:
        gate_params = self.model.get('gate_encoder', {})
        return GATEEncoderConfig(
            num_heads=int(gate_params.get('num_heads', 8)),
            hidden_dim=int(gate_params.get('hidden_dim', 256)),
            num_layers=int(gate_params.get('num_layers', 2)),
            dropout=float(gate_params.get('dropout', 0.1))
        )

    def get_sie_encoder_config(self) -> SIEEncoderConfig:
        sie_params = self.model.get('sie_encoder', {})
        return SIEEncoderConfig(
            num_patterns=int(sie_params.get('num_patterns', 16)),
            hidden_dim=int(sie_params.get('hidden_dim', 256)),
            dropout=float(sie_params.get('dropout', 0.1))
        )

    def get_ddi_predictor_config(self) -> DDIPredictorConfig:
        ddi_params = self.model.get('ddi_predictor', {})
        return DDIPredictorConfig(
            hidden_dim=int(ddi_params.get('hidden_dim', 256)),
            num_layers=int(ddi_params.get('num_layers', 3)),
            dropout=float(ddi_params.get('dropout', 0.2))
        )

    def get_data_config(self) -> DataConfig:
        data_params = self.data
        return DataConfig(
            drugbank_path=str(data_params.get('drugbank_path', 'data/drugbank/')),
            batch_size=int(data_params.get('batch_size', 32)),
            num_workers=int(data_params.get('num_workers', 4)),
            max_atoms=int(data_params.get('max_atoms', 150)),
            max_mol_weight=float(data_params.get('max_mol_weight', 2000.0))
        )

    def get_training_config(self) -> TrainingConfig:
        train_params = self.training
        return TrainingConfig(
            epochs=int(train_params.get('epochs', 100)),
            learning_rate=float(train_params.get('learning_rate', 1e-4)),
            weight_decay=float(train_params.get('weight_decay', 1e-5)),
            warmup_steps=int(train_params.get('warmup_steps', 1000)),
            gradient_clip=float(train_params.get('gradient_clip', 1.0)),
            early_stopping_patience=int(train_params.get('early_stopping_patience', 10))
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_params = self.evaluation
        return EvaluationConfig(
            metrics=eval_params.get('metrics', ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]),
            test_split=float(eval_params.get('test_split', 0.2)),
            val_split=float(eval_params.get('val_split', 0.1))
        )


def get_default_config() -> ModelConfig:
    """Load default configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
    return ModelConfig.from_yaml(config_path)