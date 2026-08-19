# modules/pooling.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GlobalMeanPool(nn.Module):
    """Global mean pooling over nodes"""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, num_nodes, hidden_dim]
        mask: [batch_size, num_nodes] (optional)
        """
        if mask is not None:
            # Apply mask
            mask = mask.unsqueeze(-1).float()
            x = x * mask
            sum_x = x.sum(dim=1)
            count = mask.sum(dim=1) + 1e-6
            return sum_x / count
        else:
            return x.mean(dim=1)


class GlobalMaxPool(nn.Module):
    """Global max pooling over nodes"""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, num_nodes, hidden_dim]
        mask: [batch_size, num_nodes] (optional)
        """
        if mask is not None:
            # Apply mask by setting masked positions to -inf
            mask = mask.unsqueeze(-1).float()
            x = x * mask + (1 - mask) * (-1e9)

        return x.max(dim=1)[0]


class GlobalAttentionPool(nn.Module):
    """Global attention pooling"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, num_nodes, hidden_dim]
        mask: [batch_size, num_nodes] (optional)
        """
        # Compute attention scores
        scores = self.attn(x)  # [batch_size, num_nodes, 1]

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)

        # Weighted sum
        out = (x * attn_weights).sum(dim=1)

        return out


class Set2Set(nn.Module):
    """Set2Set pooling from 'Order Matters' paper"""

    def __init__(
            self,
            hidden_dim: int,
            num_iters: int = 3,
            num_layers: int = 1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_iters = num_iters
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, num_nodes, hidden_dim]
        """
        batch_size = x.size(0)

        # Initial state
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        # Initial readout
        q = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for _ in range(self.num_iters):
            # Compute attention
            q_expanded = q.unsqueeze(1).expand(-1, x.size(1), -1)

            # Dot product attention
            scores = (x * q_expanded).sum(dim=-1, keepdim=True)  # [batch_size, num_nodes, 1]
            attn_weights = F.softmax(scores, dim=1)

            # Weighted sum
            r = (x * attn_weights).sum(dim=1)  # [batch_size, hidden_dim]

            # Update LSTM
            q, (h, c) = self.lstm(
                torch.cat([q, r], dim=1).unsqueeze(1),
                (h, c)
            )
            q = q.squeeze(1)

        return q


class DiffPool(nn.Module):
    """Differentiable Pooling"""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            num_clusters: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_clusters = num_clusters

        self.embed = nn.Linear(in_dim, out_dim)
        self.assign = nn.Linear(in_dim, num_clusters)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, num_nodes, in_dim]
        adj: [batch_size, num_nodes, num_nodes]
        """

        # Embedding
        embedded = self.embed(x)  # [batch_size, num_nodes, out_dim]

        # Assignment matrix
        assign = self.assign(x)  # [batch_size, num_nodes, num_clusters]

        if mask is not None:
            assign = assign.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        assign = F.softmax(assign, dim=-1)
        assign = self.dropout(assign)

        # Pooled features
        pooled_x = torch.bmm(assign.transpose(1, 2), embedded)  # [batch_size, num_clusters, out_dim]

        # Pooled adjacency
        pooled_adj = torch.bmm(
            torch.bmm(assign.transpose(1, 2), adj),
            assign
        )  # [batch_size, num_clusters, num_clusters]

        return pooled_x, pooled_adj