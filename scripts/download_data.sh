#!/bin/bash

# Download DrugBank data
echo "Downloading DrugBank data..."

# Create data directory
mkdir -p data/drugbank

# Download DrugBank vocabulary
echo "Downloading DrugBank vocabulary..."
wget -O data/drugbank/drugbank_vocabulary.csv https://go.drugbank.com/releases/latest/downloads/drugbank_vocabulary.csv

# Download drug links
echo "Downloading drug links..."
wget -O data/drugbank/drug_links.csv https://go.drugbank.com/releases/latest/downloads/drug_links.csv

# Download structures
echo "Downloading structures..."
wget -O data/drugbank/drugbank_all_structures.sdf https://go.drugbank.com/releases/latest/downloads/structures.sdf

# Download target polypeptide sequences
echo "Downloading target polypeptide sequences..."
wget -O data/drugbank/drugbank_all_target_polypeptide_sequences.fasta https://go.drugbank.com/releases/latest/downloads/target_polypeptide_sequences.fasta

# Download target polypeptide ids
echo "Downloading target polypeptide IDs..."
wget -O data/drugbank/drugbank_all_target_polypeptide_ids_all.csv https://go.drugbank.com/releases/latest/downloads/target_polypeptide_ids_all.csv

# Download pharmacologically active targets
echo "Downloading pharmacologically active targets..."
wget -O data/drugbank/drugbank_all_target_polypeptide_ids_pharmacologically_active.csv https://go.drugbank.com/releases/latest/downloads/target_polypeptide_ids_pharmacologically_active.csv

# Note about DDI data
echo ""
echo "================================================"
echo "Download complete!"
echo ""
echo "Note: DDI (Drug-Drug Interaction) data may need to be obtained from:"
echo "  - DrugBank interactions section"
echo "  - Other sources like:"
echo "    * DDI corpus (https://github.com/kilicogluh/DDI-DrugBank)"
echo "    * TWOSIDES (http://tatonettilab.org/off-sides/)"
echo "    * Other DDI databases"
echo ""
echo "Make sure to place DDI data in: data/drugbank/ddi_data.csv"
echo "================================================"