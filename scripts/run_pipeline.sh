# scripts/run_pipeline.sh

#!/bin/bash

# GANNDDI Full Pipeline Script
# This script runs the entire GANNDDI pipeline from data download to training

set -e  # Exit on error

echo "=========================================="
echo "GANNDDI Pipeline"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Setup
echo -e "\n${GREEN}Step 1: Setting up environment${NC}"
pip install -e .
pip install -r requirements.txt

# Step 2: Download data
echo -e "\n${GREEN}Step 2: Downloading DrugBank data${NC}"
bash scripts/download_data.sh

# Check if DDI data exists
if [ ! -f "data/drugbank/ddi_data.csv" ]; then
    echo -e "${YELLOW}Warning: ddi_data.csv not found.${NC}"
    echo "Please place your DDI data in data/drugbank/ddi_data.csv"
    echo "Or run: python scripts/create_dummy_ddi.py"
    read -p "Continue without DDI data? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 3: Preprocess data
echo -e "\n${GREEN}Step 3: Preprocessing data${NC}"
python scripts/preprocess_drugbank.py

# Step 4: Train model
echo -e "\n${GREEN}Step 4: Training model${NC}"
python scripts/train_model.py --wandb

# Step 5: Evaluate
echo -e "\n${GREEN}Step 5: Evaluation complete${NC}"
echo "Check checkpoints/test_results.json for test results"

echo -e "\n${GREEN}Pipeline completed successfully!${NC}"