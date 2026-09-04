# scripts/check_interaction_types.py
import pandas as pd
import json

# Load your data
train = pd.read_csv('data/preprocessed/train.csv')
print(f"Unique labels in train: {train['label'].unique()}")
print(f"Label counts: {train['label'].value_counts()}")

# Load label map
with open('data/preprocessed/label_map.json', 'r') as f:
    label_map = json.load(f)
print(f"Label map: {label_map}")