# modules/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class GraphConvLayer(nn.Module):
    """Graph Convolution Layer"""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            activation: str = 'relu',
            dropout: float = 0.0
    ):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def _get_activation(self, name: str):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif name == 'elu':
            return nn.ELU()
        else:
            return nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: node features [batch_size, num_nodes, in_dim]
        adj: adjacency matrix [batch_size, num_nodes, num_nodes]
        """

        # Message passing
        # x_agg = adj @ x
        x_agg = torch.bmm(adj, x)

        # Linear transformation
        x_out = self.linear(x_agg)

        # Activation and dropout
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)

        # Residual connection with normalization
        if x.shape[-1] == x_out.shape[-1]:
            x_out = self.norm(x + x_out)
        else:
            x_out = self.norm(x_out)

        return x_out


class MLP(nn.Module):
    """Multi-layer Perceptron"""

    def __init__(
            self,
            in_dim: int,
            hidden_dims: List[int],
            out_dim: int,
            activation: str = 'relu',
            dropout: float = 0.0,
            batch_norm: bool = True,
            residual: bool = False
    ):
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))

        self.layers = nn.Sequential(*layers)
        self.residual = residual
        self.residual_proj = nn.Linear(in_dim, out_dim) if residual and in_dim != out_dim else None

    def _get_activation(self, name: str):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif name == 'elu':
            return nn.ELU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)

        if self.residual:
            if self.residual_proj is not None:
                x = self.residual_proj(x)
            out = out + x

        return out


class ResidualBlock(nn.Module):
    """Residual Block with optional bottleneck"""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bottleneck_dim: Optional[int] = None,
            dropout: float = 0.0,
            activation: str = 'relu'
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bottleneck_dim = bottleneck_dim or out_dim // 4

        if bottleneck_dim:
            self.block = nn.Sequential(
                nn.Linear(in_dim, self.bottleneck_dim),
                self._get_activation(activation),
                nn.Linear(self.bottleneck_dim, out_dim)
            )
        else:
            self.block = nn.Linear(in_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation(activation)

        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def _get_activation(self, name: str):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.block(x)
        out = self.dropout(out)
        out = self.norm(out + residual)
        out = self.activation(out)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for graph nodes"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch_size, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1), :]


class LayerNorm(nn.Module):
    """Layer Normalization with optional bias"""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output