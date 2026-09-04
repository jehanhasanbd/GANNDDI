Here are all the commands to run the GANNDDI system:

## 1. Initial Setup Commands

```bash
# Clone or navigate to project directory
cd gannddi

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and dependencies
pip install -e .
pip install -r requirements.txt

# Install additional development dependencies (optional)
pip install pytest pytest-cov black flake8 isort pre-commit wandb optuna
```

## 2. Data Preparation Commands

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Download DrugBank data
bash scripts/download_data.sh

# If you don't have DDI data, create dummy data for testing
python scripts/create_dummy_ddi.py

# Preprocess the data
python scripts/preprocess_drugbank.py

# Preprocess with custom parameters
python scripts/preprocess_drugbank.py \
    --data_dir data/drugbank/ \
    --output_dir data/preprocessed/ \
    --max_atoms 150 \
    --max_mol_weight 2000.0 \
    --test_split 0.2 \
    --val_split 0.1 \
    --seed 42
```

## 3. Training Commands

```bash
# Basic training with default config
python scripts/train_model.py

# Training with custom config file
python scripts/train_model.py --config config/default.yaml

# Training with specific parameters
python scripts/train_model.py \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --num_heads 8 \
    --num_patterns 16 \
    --dropout 0.2

# Training with GPU
python scripts/train_model.py --gpu 0

# Training with wandb logging
python scripts/train_model.py --wandb

# Training with wandb and custom project/run name
python scripts/train_model.py \
    --wandb \
    --wandb_project gannddi \
    --wandb_run_name experiment_1

# Resume training from checkpoint
python scripts/train_model.py --resume checkpoints/best_model.pt

# Full training with all options
python scripts/train_model.py \
    --config config/default.yaml \
    --data_dir data/preprocessed/ \
    --output_dir checkpoints/ \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --num_heads 8 \
    --num_patterns 16 \
    --dropout 0.2 \
    --wandb \
    --gpu 0 \
    --seed 42
```

## 4. Evaluation Commands

```bash
# Evaluate on test set
python scripts/evaluate_model.py \
    --model_path checkpoints/best_model.pt \
    --data_dir data/preprocessed/ \
    --split test

# Evaluate on validation set
python scripts/evaluate_model.py \
    --model_path checkpoints/best_model.pt \
    --data_dir data/preprocessed/ \
    --split val

# Evaluate with predictions saved
python scripts/evaluate_model.py \
    --model_path checkpoints/best_model.pt \
    --data_dir data/preprocessed/ \
    --split test \
    --save_predictions \
    --output_dir evaluation/
```

## 5. Inference Commands

```bash
# Predict a single drug pair
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --input "CC(=O)OC1=CC=CC=C1C(=O)O,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

# Predict from CSV file
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --input data/sample_pairs.csv \
    --output predictions.csv

# Batch inference with GPU
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --input data/sample_pairs.csv \
    --output predictions.csv \
    --batch_size 64 \
    --gpu 0
```

## 6. Hyperparameter Tuning Commands

```bash
# Run hyperparameter tuning with Optuna
python experiments/hyperparameter_tuning.py

# Custom tuning with more trials
python -c "
from gannddi.experiments import HyperparameterTuner
from gannddi.config import get_default_config

config = get_default_config()
tuner = HyperparameterTuner(config.model.__dict__, n_trials=50)
results = tuner.run_optuna_tuning()
"
```

## 7. Testing Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v
pytest tests/test_data.py -v
pytest tests/test_utils.py -v

# Run tests with coverage
pytest tests/ -v --cov=gannddi --cov-report=html

# Run tests with coverage in terminal
pytest tests/ -v --cov=gannddi --cov-report=term

# Run linting
flake8 gannddi tests
black --check gannddi tests
isort --check-only gannddi tests

# Auto-fix linting issues
black gannddi tests
isort gannddi tests
```

## 8. Running the Full Pipeline

```bash
# Run the entire pipeline (setup, data, training)
bash scripts/run_pipeline.sh

# Or run individual steps manually
pip install -e .
pip install -r requirements.txt
bash scripts/download_data.sh
python scripts/preprocess_drugbank.py
python scripts/train_model.py --wandb
```

## 9. Running with Docker

```bash
# Build Docker image
docker build -t gannddi:latest .

# Run with Docker (CPU)
docker run --rm -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    gannddi:latest \
    python scripts/train_model.py

# Run with Docker (GPU)
docker run --rm -it \
    --runtime=nvidia \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    gannddi:latest \
    python scripts/train_model.py --gpu 0

# Run with docker-compose
docker-compose up
```

## 10. Jupyter Notebook Commands

```bash
# Start Jupyter notebook
jupyter notebook notebooks/eda.ipynb

# Start with specific port
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

# Open visualization demo
jupyter notebook notebooks/visualization_demo.ipynb
```

## 11. TensorBoard Commands

```bash
# Start TensorBoard
tensorboard --logdir=logs/ --port=6006

# Start TensorBoard with specific host
tensorboard --logdir=logs/ --host=0.0.0.0 --port=6006
```

## 12. Monitoring Commands

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/training.log

# Check model checkpoints
ls -la checkpoints/

# View training history
cat checkpoints/training_history.json

# View test results
cat checkpoints/test_results.json

# View best model info
cat checkpoints/best_model.pt
```

## 13. Quick One-Line Commands

```bash
# Complete setup and training (if data is already downloaded)
pip install -e . && pip install -r requirements.txt && python scripts/preprocess_drugbank.py && python scripts/train_model.py

# Quick training with default settings
python scripts/train_model.py --epochs 10

# Quick evaluation
python scripts/evaluate_model.py --model_path checkpoints/best_model.pt

# Quick inference
python scripts/inference.py --model_path checkpoints/best_model.pt --input "CC(=O)OC1=CC=CC=C1C(=O)O,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
```

## 14. Debugging Commands

```bash
# Check Python version
python --version

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check CUDA devices
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Check RDKit installation
python -c "from rdkit import Chem; print('RDKit OK')"

# Check all imports
python -c "
from gannddi.config import get_default_config
from gannddi.data import DrugBankLoader
from gannddi.models import DDIPredictor
from gannddi.training import Trainer
from gannddi.utils import MetricsCalculator
print('All imports OK')
"

# Run minimal system test
python -c "
import torch
import numpy as np
from gannddi.data import MolecularGraph

smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
graph = MolecularGraph.from_smiles(smiles, 'DB00001')
print(f'Graph created: {graph.num_atoms} atoms, {graph.num_bonds} bonds')
print('System ready')
"
```

## 15. Cleanup Commands

```bash
# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

# Remove data (caution!)
rm -rf data/drugbank/
rm -rf data/preprocessed/

# Remove checkpoints
rm -rf checkpoints/

# Remove logs
rm -rf logs/

# Remove evaluation results
rm -rf evaluation/

# Remove build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf .pytest_cache/
rm -rf .coverage
rm -rf htmlcov/

# Remove Docker images
docker rmi gannddi:latest
docker system prune -a

# Full clean (use make if available)
make clean
```

## 16. Environment Variables

```bash
# Set environment variables for current session
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_API_KEY=your_wandb_api_key
export WANDB_PROJECT=gannddi

# Or create .env file
echo "CUDA_VISIBLE_DEVICES=0" > .env
echo "PYTHONPATH=${PYTHONPATH}:$(pwd)" >> .env
echo "WANDB_API_KEY=your_wandb_api_key" >> .env

# Load .env file
source .env
```

## 17. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

## 18. Export Model for Production

```bash
# Export model to TorchScript
python -c "
import torch
from gannddi.models import DDIPredictor

checkpoint = torch.load('checkpoints/best_model.pt')
model = DDIPredictor(num_classes=65)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Script the model
scripted_model = torch.jit.script(model)
scripted_model.save('gannddi_production.pt')
print('Model exported to gannddi_production.pt')
"

# Run inference with exported model
python scripts/inference.py --model_path gannddi_production.pt --input "SMILES1,SMILES2"
```

## 19. CI/CD Commands (GitHub Actions)

```bash
# Run CI checks locally (if using act)
act -j test

# Or run GitHub Actions locally with act
act --container-architecture linux/amd64
```

## 20. Custom Experiment Commands

```bash
# Create custom configuration
cp config/default.yaml config/my_experiment.yaml

# Edit config
vim config/my_experiment.yaml

# Run experiment with custom config
python experiments/run_experiment.py --config config/my_experiment.yaml --wandb

# Run multiple experiments
for seed in 42 43 44 45; do
    python scripts/train_model.py --seed $seed --wandb_run_name "seed_${seed}"
done
```

## Quick Reference Card

| Task | Command |
|------|---------|
| Install | `pip install -e . && pip install -r requirements.txt` |
| Download Data | `bash scripts/download_data.sh` |
| Preprocess | `python scripts/preprocess_drugbank.py` |
| Train | `python scripts/train_model.py` |
| Evaluate | `python scripts/evaluate_model.py --model_path checkpoints/best_model.pt` |
| Inference | `python scripts/inference.py --model_path checkpoints/best_model.pt --input "SMILES1,SMILES2"` |
| Test | `pytest tests/ -v` |
| Lint | `black gannddi tests && isort gannddi tests && flake8 gannddi tests` |
| Clean | `find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true` |
| Docker | `docker build -t gannddi . && docker run --rm -it gannddi` |
| GPU | `python scripts/train_model.py --gpu 0` |
| WandB | `python scripts/train_model.py --wandb` |


Here's the complete code structure tree for your GANNDDI project:



```angular2html
GANNDDI/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── default.yaml
│   └── model_config.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── drugbank_loader.py
│   ├── preprocessing.py
│   └── molecular_graph.py
│
├── models/
│   ├── __init__.py
│   ├── ddi_predictor.py
│   ├── gaan.py
│   ├── gate_encoder.py
│   ├── sie_encoder.py
│   └── attention.py
│
├── modules/
│   ├── __init__.py
│   ├── layers.py
│   ├── gates.py
│   ├── multi_head_attention.py
│   └── pooling.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── chem_utils.py
│   └── data_utils.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── loss_functions.py
│
├── experiments/
│   ├── __init__.py
│   ├── run_experiment.py
│   └── hyperparameter_tuning.py
│
├── scripts/
│   ├── download_data.sh
│   ├── preprocess_drugbank.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── inference.py
│   ├── create_multiclass_data.py
│   ├── add_negative_samples.py
│   ├── check_interaction_types.py
│   └── run_pipeline.sh
│
├── notebooks/
│   ├── eda.ipynb
│   └── visualization_demo.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_utils.py
│   └── test_training.py
│
├── checkpoints/
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── training_history.json
│   └── test_results.json
│
├── evaluation/
│   ├── metrics_test.json
│   ├── predictions_test.csv
│   ├── classification_report_test.txt
│   └── confusion_matrix_test.png
│
├── data/
│   ├── drugbank/
│   │   ├── drug_links.csv
│   │   ├── drugbank_vocabulary.csv
│   │   ├── drugbank_all_structures.sdf
│   │   ├── drugbank_all_target_polypeptide_sequences.fasta
│   │   ├── drugbank_all_target_polypeptide_ids_all.csv
│   │   ├── drugbank_all_target_polypeptide_ids_pharmacologically_active.csv
│   │   ├── drugbank_all_drug_sequences.fasta
│   │   └── ddi_data.csv
│   │
│   └── preprocessed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── drug_smiles.json
│       ├── drug_info.json
│       ├── label_map.json
│       └── stats.json
│
├── logs/
│   └── training.log
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .pre-commit-config.yaml
│
└── .github/
    └── workflows/
        └── ci.yml
```

## Detailed Description of Each Directory

### Root Directory
| File | Purpose |
|------|---------|
| `README.md` | Project documentation and setup guide |
| `requirements.txt` | Python dependencies |
| `setup.py` | Package installation configuration |
| `.gitignore` | Git ignore rules |
| `Dockerfile` | Docker container configuration |
| `docker-compose.yml` | Multi-container Docker setup |
| `Makefile` | Common development commands |
| `.pre-commit-config.yaml` | Pre-commit hooks for code quality |

### config/
| File | Purpose |
|------|---------|
| `__init__.py` | Config module initialization |
| `default.yaml` | Default model and training configuration |
| `model_config.py` | Configuration dataclasses and loading |

### data/
| File | Purpose |
|------|---------|
| `__init__.py` | Data module initialization |
| `dataset.py` | PyTorch Dataset classes for DDI data |
| `drugbank_loader.py` | Load and preprocess DrugBank data |
| `preprocessing.py` | Data preprocessing utilities |
| `molecular_graph.py` | Molecular graph representation from SMILES |

### models/
| File | Purpose |
|------|---------|
| `__init__.py` | Models module initialization |
| `ddi_predictor.py` | Main DDI prediction model (combines GATE + SIE) |
| `gaan.py` | Gated Attention Network implementation |
| `gate_encoder.py` | GATE Encoder for drug interactions |
| `sie_encoder.py` | SIE Encoder for similarity identification |
| `attention.py` | Multi-head attention modules |

### modules/
| File | Purpose |
|------|---------|
| `__init__.py` | Modules initialization |
| `layers.py` | Basic neural network layers |
| `gates.py` | Gating mechanisms (GLU, Gated Attention, etc.) |
| `multi_head_attention.py` | Multi-head attention variants |
| `pooling.py` | Graph pooling operations |

### utils/
| File | Purpose |
|------|---------|
| `__init__.py` | Utils initialization |
| `metrics.py` | Evaluation metrics (accuracy, F1, AUROC, etc.) |
| `visualization.py` | Plotting and visualization utilities |
| `chem_utils.py` | Chemistry utilities (SMILES validation, fingerprints) |
| `data_utils.py` | General data utilities |

### training/
| File | Purpose |
|------|---------|
| `__init__.py` | Training module initialization |
| `trainer.py` | Main training loop with early stopping |
| `evaluator.py` | Model evaluation utilities |
| `loss_functions.py` | Custom loss functions (Focal Loss, Label Smoothing, etc.) |

### experiments/
| File | Purpose |
|------|---------|
| `__init__.py` | Experiments initialization |
| `run_experiment.py` | Run full experiments with logging |
| `hyperparameter_tuning.py` | Optuna-based hyperparameter optimization |

### scripts/
| File | Purpose |
|------|---------|
| `download_data.sh` | Download DrugBank data files |
| `preprocess_drugbank.py` | Preprocess DrugBank data |
| `train_model.py` | Main training script |
| `evaluate_model.py` | Evaluate trained models |
| `inference.py` | Run inference on new drug pairs |
| `create_multiclass_data.py` | Create multi-class dataset |
| `add_negative_samples.py` | Add negative samples for binary classification |
| `check_interaction_types.py` | Check available interaction types |
| `run_pipeline.sh` | Run full pipeline (download → preprocess → train) |

### notebooks/
| File | Purpose |
|------|---------|
| `eda.ipynb` | Exploratory Data Analysis |
| `visualization_demo.ipynb` | Visualization examples |

### tests/
| File | Purpose |
|------|---------|
| `__init__.py` | Tests initialization |
| `test_models.py` | Model architecture tests |
| `test_data.py` | Data loading and preprocessing tests |
| `test_utils.py` | Utility function tests |
| `test_training.py` | Training and evaluation tests |

### data/drugbank/
| File | Purpose |
|------|---------|
| `drug_links.csv` | Drug links and metadata |
| `drugbank_vocabulary.csv` | Drug vocabulary |
| `drugbank_all_structures.sdf` | Molecular structures (SDF format) |
| `drugbank_all_target_polypeptide_sequences.fasta` | Target protein sequences |
| `drugbank_all_target_polypeptide_ids_all.csv` | All target polypeptide IDs |
| `drugbank_all_target_polypeptide_ids_pharmacologically_active.csv` | Active targets |
| `drugbank_all_drug_sequences.fasta` | Drug sequences |
| `ddi_data.csv` | Drug-drug interaction data |

### data/preprocessed/
| File | Purpose |
|------|---------|
| `train.csv` | Training dataset |
| `val.csv` | Validation dataset |
| `test.csv` | Test dataset |
| `drug_smiles.json` | Drug ID to SMILES mapping |
| `drug_info.json` | Drug metadata |
| `label_map.json` | Label to class mapping |
| `stats.json` | Dataset statistics |

### checkpoints/
| File | Purpose |
|------|---------|
| `best_model.pt` | Best performing model checkpoint |
| `final_model.pt` | Final model checkpoint |
| `training_history.json` | Training metrics history |
| `test_results.json` | Test evaluation results |

### evaluation/
| File | Purpose |
|------|---------|
| `metrics_test.json` | Test metrics |
| `predictions_test.csv` | Test predictions |
| `classification_report_test.txt` | Classification report |
| `confusion_matrix_test.png` | Confusion matrix visualization |

### .github/workflows/
| File | Purpose |
|------|---------|
| `ci.yml` | GitHub Actions CI/CD pipeline |

## File Count Summary

| Directory | Files |
|-----------|-------|
| config/ | 3 |
| data/ | 5 |
| models/ | 6 |
| modules/ | 5 |
| utils/ | 5 |
| training/ | 4 |
| experiments/ | 3 |
| scripts/ | 9 |
| notebooks/ | 2 |
| tests/ | 5 |
| Root | 8 |
| **Total** | **~55 Python files** |