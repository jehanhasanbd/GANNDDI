"""
Gated Attention Network for molecular graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class GatedAttentionLayer(nn.Module):
    """Gated Attention Network layer with multi-head attention and gates"""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            gate_dim: int = 16,
            dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gate_dim = gate_dim
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, num_heads),
            nn.Sigmoid()
        )

        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if edge_index.numel() == 0 or x.size(0) == 0:
            return x

        num_nodes = x.size(0)

        # Get dtype and device from x
        dtype = x.dtype
        device = x.device

        # Linear projections - keep same dtype as x
        Q = self.query(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.key(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.value(x).view(num_nodes, self.num_heads, self.head_dim)

        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]

        # Compute attention scores
        q_src = Q[src_nodes]
        k_tgt = K[tgt_nodes]
        scores = torch.sum(q_src * k_tgt, dim=-1) * self.scale

        # Softmax per target node
        attention_weights = torch.zeros_like(scores)
        unique_targets = torch.unique(tgt_nodes)
        for target in unique_targets:
            mask = (tgt_nodes == target)
            if mask.sum() > 0:
                target_scores = scores[mask]
                target_weights = F.softmax(target_scores, dim=0)
                attention_weights[mask] = target_weights

        # Compute gates
        max_neighbors = torch.zeros(num_nodes, self.hidden_dim, dtype=dtype, device=device)
        mean_neighbors = torch.zeros(num_nodes, self.hidden_dim, dtype=dtype, device=device)
        neighbor_counts = torch.zeros(num_nodes, dtype=dtype, device=device)

        for i, (src, tgt) in enumerate(zip(src_nodes.tolist(), tgt_nodes.tolist())):
            src_feat = x[src]
            max_neighbors[tgt] = torch.max(max_neighbors[tgt], src_feat)
            mean_neighbors[tgt] += src_feat
            neighbor_counts[tgt] += 1

        neighbor_counts = torch.clamp(neighbor_counts, min=1)
        mean_neighbors = mean_neighbors / neighbor_counts.unsqueeze(1)

        # Compute gate values
        gate_input = torch.cat([
            x,
            max_neighbors,
            mean_neighbors[:, :16]
        ], dim=-1)

        gates = self.gate_mlp(gate_input)  # [num_nodes, num_heads]
        gates = gates.unsqueeze(2)  # [num_nodes, num_heads, 1]

        # Aggregate messages
        messages = torch.zeros(num_nodes, self.num_heads, self.head_dim, dtype=dtype, device=device)

        for i, (src, tgt) in enumerate(zip(src_nodes.tolist(), tgt_nodes.tolist())):
            msg = V[src] * attention_weights[i].unsqueeze(1) * gates[tgt]
            messages[tgt] += msg

        # Reshape and apply output projection
        messages = messages.view(num_nodes, -1)
        output = self.norm(x + self.dropout(self.out(messages)))

        return output


class GatedAttentionNetwork(nn.Module):
    """Gated Attention Network for molecular graphs"""

    def __init__(
        self,
        node_feat_dim: Optional[int] = None,
        edge_feat_dim: int = 10,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        gate_dim: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.node_embed = None
        self.edge_embed = None

        self.layers = nn.ModuleList([
            GatedAttentionLayer(hidden_dim, num_heads, gate_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _init_embeddings(self, node_feat_dim: int, edge_feat_dim: int):
        device = next(self.layers[0].parameters()).device

        if self.node_embed is None or self.node_embed.in_features != node_feat_dim:
            self.node_embed = nn.Linear(node_feat_dim, self.hidden_dim).to(device)

        if edge_feat_dim > 0 and (self.edge_embed is None or self.edge_embed.in_features != edge_feat_dim):
            self.edge_embed = nn.Linear(edge_feat_dim, self.hidden_dim // 4).to(device)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        batch_size = node_features.size(0)
        max_nodes = node_features.size(1)
        node_feat_dim = node_features.size(2)

        self._init_embeddings(node_feat_dim, self.edge_feat_dim)

        # Flatten node features: [batch_size * max_nodes, feat_dim]
        x = node_features.view(-1, node_feat_dim)

        # Find valid nodes (non-padding)
        valid_mask = (x.abs().sum(dim=1) > 0)
        valid_indices = valid_mask.nonzero().squeeze(1)

        if len(valid_indices) == 0:
            return torch.zeros(batch_size, self.output_dim, device=node_features.device)

        # Get valid node features
        x_valid = x[valid_indices]

        # Process edges
        flat_edge_index = torch.zeros(2, 0, dtype=torch.long, device=node_features.device)
        edge_feat_flat = None

        if edge_index is not None and edge_index.numel() > 0:
            # Flatten edge indices: [2, batch_size * max_edges]
            flat_edge_index = edge_index.view(2, -1)

            # Remove padding edges
            valid_edges = (flat_edge_index[0] != 0) | (flat_edge_index[1] != 0)
            valid_edges_indices = valid_edges.nonzero().squeeze(1)

            if valid_edges_indices.numel() > 0:
                # Get valid edges
                flat_edge_index = flat_edge_index[:, valid_edges_indices]

                # Determine which graph each edge belongs to
                edge_graph_idx = flat_edge_index[0] // max_nodes

                # Remove graph offset from node indices
                flat_edge_index[0] = flat_edge_index[0] - edge_graph_idx * max_nodes
                flat_edge_index[1] = flat_edge_index[1] - edge_graph_idx * max_nodes

                # Create mapping from original to compressed indices
                original_to_compressed = torch.full(
                    (batch_size * max_nodes,), -1,
                    dtype=torch.long, device=node_features.device
                )
                original_to_compressed[valid_indices] = torch.arange(
                    len(valid_indices), device=node_features.device
                )

                # Map edges to compressed indices
                edge_src_original = flat_edge_index[0] + edge_graph_idx * max_nodes
                edge_tgt_original = flat_edge_index[1] + edge_graph_idx * max_nodes

                edge_src_compressed = original_to_compressed[edge_src_original]
                edge_tgt_compressed = original_to_compressed[edge_tgt_original]

                # Remove invalid edges
                valid_edges_final = (edge_src_compressed >= 0) & (edge_tgt_compressed >= 0)
                valid_edges_final_indices = valid_edges_final.nonzero().squeeze(1)

                if valid_edges_final_indices.numel() > 0:
                    flat_edge_index = torch.stack([
                        edge_src_compressed[valid_edges_final_indices],
                        edge_tgt_compressed[valid_edges_final_indices]
                    ], dim=0)

                    # Process edge features
                    if edge_features is not None and self.edge_embed is not None and edge_features.numel() > 0:
                        if edge_features.dim() == 3:
                            flat_edge_features = edge_features.view(-1, self.edge_feat_dim)
                        else:
                            flat_edge_features = edge_features

                        # Apply filters
                        if valid_edges_indices.numel() > 0 and valid_edges_indices.max() < flat_edge_features.size(0):
                            flat_edge_features = flat_edge_features[valid_edges_indices]

                        if valid_edges_final_indices.numel() > 0 and valid_edges_final_indices.max() < flat_edge_features.size(0):
                            flat_edge_features = flat_edge_features[valid_edges_final_indices]

                        if flat_edge_features.size(0) > 0 and flat_edge_features.size(0) == flat_edge_index.size(1):
                            edge_feat_flat = self.edge_embed(flat_edge_features)
                else:
                    flat_edge_index = torch.zeros(2, 0, dtype=torch.long, device=node_features.device)
            else:
                flat_edge_index = torch.zeros(2, 0, dtype=torch.long, device=node_features.device)

        # Embed nodes
        x = self.node_embed(x_valid)
        x = self.dropout(x)

        # Apply Gated Attention layers
        layer_outputs = [x]
        for layer in self.layers:
            x = layer(x, flat_edge_index, batch, edge_feat_flat)
            layer_outputs.append(x)

        # Create batch mapping for pooling
        if len(valid_indices) > 0:
            batch_idx = valid_indices // max_nodes
        else:
            batch_idx = torch.zeros(len(x), dtype=torch.long, device=node_features.device)

        # Mean pooling
        if len(x) > 0 and len(torch.unique(batch_idx)) > 0:
            x_pooled = scatter_mean(x, batch_idx, dim=0)

            # Ensure batch_size outputs
            if x_pooled.size(0) < batch_size:
                padding = torch.zeros(batch_size - x_pooled.size(0), self.hidden_dim, device=x.device)
                x_pooled = torch.cat([x_pooled, padding], dim=0)
            elif x_pooled.size(0) > batch_size:
                x_pooled = x_pooled[:batch_size]
        else:
            x_pooled = torch.zeros(batch_size, self.hidden_dim, device=node_features.device)

        output = self.output_proj(x_pooled)

        if return_all_layers:
            return output, layer_outputs

        return output


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Scatter mean implementation"""
    if src.dim() > 2:
        src = src.view(src.size(0), -1)

    if index.numel() == 0:
        return torch.zeros(0, src.size(1), dtype=src.dtype, device=src.device)

    # Ensure index is valid
    max_idx = index.max().item()
    num_indices = max_idx + 1

    if num_indices == 0:
        return torch.zeros(0, src.size(1), dtype=src.dtype, device=src.device)

    # Use the same dtype as src - FIXED: match dtype
    counts = torch.bincount(index, minlength=num_indices)
    counts = counts.float().to(src.dtype).unsqueeze(1).to(src.device)

    # Create summed tensor with same dtype as src
    summed = torch.zeros(num_indices, src.size(1), dtype=src.dtype, device=src.device)
    summed.index_add_(0, index, src)

    # Avoid division by zero
    counts = torch.clamp(counts, min=1)

    return summed / counts