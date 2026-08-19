# modules/gates.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU)"""

    def __init__(self, in_dim: int, out_dim: int, activation: str = 'sigmoid'):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str):
        if name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        else:
            return nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, ..., in_dim]
        Returns: [batch_size, ..., out_dim]
        """
        out = self.linear(x)
        a, b = out.chunk(2, dim=-1)
        return a * self.activation(b)


class GatedAttention(nn.Module):
    """Gated Attention module"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 1,
            gate_dim: int = 16,
            dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Gate computation
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, gate_dim),
            nn.Sigmoid()
        )

        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, num_nodes, hidden_dim]
        context: [batch_size, num_nodes, hidden_dim] (optional)
        Returns: [batch_size, num_nodes, hidden_dim]
        """

        if context is None:
            context = x

        # Compute query, key, value
        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)

        # Reshape for multi-head attention
        batch_size = x.size(0)
        num_nodes = x.size(1)

        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)

        # Compute gates
        gate_input = torch.cat([
            x,
            out
        ], dim=-1)

        gate = self.gate_mlp(gate_input.mean(dim=1, keepdim=True))

        # Apply gate
        out = self.out(out) * gate

        return out, gate


class GatedAggregator(nn.Module):
    """Gated Aggregator with multiple heads"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            gate_dim: int = 16,
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads

        self.gates = nn.ModuleList([
            GatedAttention(hidden_dim, 1, gate_dim, dropout)
            for _ in range(num_heads)
        ])

        self.out = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [batch_size, num_nodes, hidden_dim]
        """

        outputs = []
        gates = []

        for gate in self.gates:
            out, g = gate(x, context, mask)
            outputs.append(out)
            gates.append(g)

        # Concatenate outputs from all heads
        concat = torch.cat(outputs, dim=-1)

        # Apply output projection
        out = self.out(concat)
        out = self.dropout(out)
        out = self.norm(x + out)

        return out


class SoftGate(nn.Module):
    """Soft gating mechanism"""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return soft gates in [0, 1]"""
        logits = self.linear(x)
        return F.softmax(logits / self.temperature, dim=-1)


class SigmoidGate(nn.Module):
    """Sigmoid gating mechanism"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return gates in [0, 1]"""
        return self.sigmoid(self.linear(x))