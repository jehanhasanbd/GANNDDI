"""
Visualization utilities for GANNDDI
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors


class Visualizer:
    """Visualization utilities for model analysis"""

    @staticmethod
    def plot_training_curves(
            history: Dict,
            metrics: List[str] = ['loss', 'accuracy', 'f1'],
            save_path: Optional[str] = None
    ):
        """Plot training and validation curves"""

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            train_values = [h.get(metric, 0) for h in history.get('train', [])]
            val_values = [h.get(metric, 0) for h in history.get('val', [])]

            axes[i].plot(train_values, label='Train', linewidth=2)
            axes[i].plot(val_values, label='Validation', linewidth=2)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} Curves')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            class_names: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto'
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_attention_weights(
            attention_weights: torch.Tensor,
            titles: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """Visualize attention weights"""

        if attention_weights.dim() == 4:
            # [batch, heads, query, key]
            attention_weights = attention_weights.mean(dim=1)  # Average over heads

        n_plots = min(attention_weights.size(0), 4)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

        if n_plots == 1:
            axes = [axes]

        for i in range(n_plots):
            im = axes[i].imshow(
                attention_weights[i].detach().cpu().numpy(),
                cmap='viridis',
                aspect='auto'
            )
            axes[i].set_title(titles[i] if titles and i < len(titles) else f'Sample {i}')
            axes[i].set_xlabel('Key')
            axes[i].set_ylabel('Query')
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_molecule(
            smiles: str,
            title: Optional[str] = None,
            save_path: Optional[str] = None
    ):
        """Visualize molecule from SMILES"""

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return

        # Draw molecule
        img = Draw.MolToImage(mol, size=(400, 400))

        if title:
            # Add title using matplotlib
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(title, fontsize=14)
            ax.axis('off')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            img.show()

    @staticmethod
    def plot_molecular_properties(
            df: pd.DataFrame,
            properties: List[str] = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'],
            save_path: Optional[str] = None
    ):
        """Plot molecular properties distribution"""

        n_cols = 2
        n_rows = (len(properties) + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, prop in enumerate(properties):
            if prop in df.columns:
                axes[i].hist(df[prop].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_xlabel(prop)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Distribution of {prop}')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].set_visible(False)

        # Remove empty subplots
        for i in range(len(properties), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_feature_importance(
            feature_names: List[str],
            importance_scores: np.ndarray,
            top_k: int = 20,
            save_path: Optional[str] = None
    ):
        """Plot feature importance"""

        # Sort by importance
        indices = np.argsort(importance_scores)[::-1][:top_k]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]

        plt.figure(figsize=(10, max(6, top_k * 0.3)))
        plt.barh(top_features, top_scores)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Most Important Features')
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_embedding_tsne(
            embeddings: np.ndarray,
            labels: np.ndarray,
            class_names: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """Plot t-SNE visualization of embeddings"""

        from sklearn.manifold import TSNE

        # Reduce dimensions
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 8))

        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = class_names[i] if class_names and i < len(class_names) else f'Class {label}'
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=label_name,
                alpha=0.7
            )

        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Visualization of Embeddings')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()