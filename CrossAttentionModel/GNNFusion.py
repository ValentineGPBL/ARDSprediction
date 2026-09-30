"""
Graph-attention multimodal fusion - all pytorch.
Based on GAT(Meta)

   modality3: text, image, metadata
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional



NODE_NAMES = ["text", "image", "metadata"]



def build_adjacency() -> torch.Tensor:
    n = 3
    # adjacency with self-loops
    adj = torch.eye(n, dtype=torch.bool)

    pairs = [
        (0, 1),  # text <> image
        (0, 2),  # text <> metadata
        (1, 2),  # image <> metadata
    ]

    for i, j in pairs:
        adj[i, j] = True
        adj[j, i] = True

    return adj

class GATLayer(nn.Module):
    """Multi-head graph attention over a fixed small graph"""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.negative_slope = negative_slope

        self.W = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        #linear projection (W(embedding))
        self.a_src = nn.Parameter(torch.zeros(n_heads, self.d_head)) #hi = source node
        self.a_dst = nn.Parameter(torch.zeros(n_heads, self.d_head)) #hj = destination node
        #adds dim for (batch,head,features) and randomizes zero values
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        missing_mask = None,
    ) -> torch.Tensor:
        # GAT formula = eij = a^T[W(hi)][W(hj)] (meta)

        B, N, _ = x.shape
        h = self.W(x).view(
            B,
            N,
            self.n_heads,
            self.d_head)

        #source scores
        ei = (h * self.a_src.view(1, 1, self.n_heads, self.d_head)).sum(-1)
        #destination scores
        ej = (h * self.a_dst.view(1, 1, self.n_heads, self.d_head)).sum(-1)
        #edge scores
        e = ei.unsqueeze(2) + ej.unsqueeze(1)
        #positive > unchanged
        #negative > x slope
        e = F.leaky_relu(e, self.negative_slope)

        """edge_mask = adj.view(1, N, N, 1).to(device=x.device, dtype=torch.bool)
        pair_missing = missing_mask.unsqueeze(2) | missing_mask.unsqueeze(1)
        invalid = ~edge_mask | pair_missing.unsqueeze(-1)
        e = e.masked_fill(invalid, float("-inf"))"""

        edge_mask = adj.view(1, N, N, 1).to(device=x.device, dtype=torch.bool)

        #Only enforce graph connectivity
        #Missing modalities are represented by learned sentinel embeddings
        #remain part of the graph
        invalid = ~edge_mask

        e = e.masked_fill(invalid, float("-inf"))

        """all_invalid = invalid.all(dim=2, keepdim=True)
        e = e.masked_fill(all_invalid, 0.0)
        alpha = torch.where(all_invalid, torch.zeros_like(alpha), alpha)"""

        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        out = torch.einsum("bnmh,bmhd->bnhd", alpha, h)
        return out.reshape(B, N, self.d_model)


class PerVisitGNNFusion(nn.Module):
    """
    Per-visit multimodal fusion with stacked GAT layers and attention readout.

    Same output shape as PerVisitFusion: (B, d_model).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        modality_dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_nodes = 3
        self.modality_dropout = modality_dropout

        self.register_buffer("adjacency", build_adjacency(), persistent=False)

        self.modality_embeddings = nn.Embedding(self.n_nodes, d_model)
        self.sentinel_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            for _ in range(self.n_nodes)
        ])
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            )
            for _ in range(self.n_nodes)
        ])

        self.gat_layers = nn.ModuleList([
            GATLayer(d_model = d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.aggregation_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Missing modalities are encoded as learned sentinel tokens
    # and fully participate in attention and pooling
    def apply_sentinels(
        self,
        embeddings: List[torch.Tensor],
        missing_mask: torch.Tensor,
    ) -> List[torch.Tensor]:

        B = embeddings[0].size(0)
        device = embeddings[0].device

        processed = []
        for i, emb in enumerate(embeddings):
            sentinel = self.sentinel_tokens[i].expand(B, 1, self.d_model).squeeze(1)

            is_missing = missing_mask[:, i].unsqueeze(1).float()

            emb = emb * (1 - is_missing) + sentinel * is_missing
            processed.append(emb)

        if self.training and self.modality_dropout > 0:
            for i in range(self.n_nodes):
                drop_mask = torch.rand(B, device=device) < self.modality_dropout
                all_missing = missing_mask.all(dim=1)
                drop_mask = drop_mask & ~all_missing
                sentinel = self.sentinel_tokens[i].expand(B, 1, self.d_model).squeeze(1)
                processed[i] = torch.where(
                    drop_mask.unsqueeze(1), sentinel, processed[i]
                )
        return processed

    def encode_nodes(
        self,
        embeddings: List[torch.Tensor],
        missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        processed = self.apply_sentinels(embeddings, missing_mask)
        type_embs = self.modality_embeddings(
            torch.arange(self.n_nodes, device=processed[0].device)
        )
        tokens = []
        for i, emb in enumerate(processed):
            emb = self.input_projs[i](emb) + type_embs[i]
            tokens.append(emb.unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def GAT_stack(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        adj = self.adjacency
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            x = norm(x + gat(x, adj, missing_mask))
        return x

    def pool(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:

        """scores = self.aggregation_attn(x)
        scores = scores.masked_fill(missing_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)"""

        scores = self.aggregation_attn(x)
        weights = torch.softmax(scores, dim=1)

        fused = (x * weights).sum(dim=1)
        return self.output_norm(fused)

    def forward(
            self,
            text_emb: torch.Tensor,
            image_emb: torch.Tensor,
            metadata_emb: torch.Tensor,
            missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B = text_emb.size(0)

        if missing_mask is None:
            missing_mask = torch.zeros(B,3,dtype=torch.bool,device=text_emb.device)

        embeddings = [text_emb,image_emb,metadata_emb]
        x = self.encode_nodes(embeddings,missing_mask)
        x = self.GAT_stack(x,missing_mask)

        return self.pool(x, missing_mask)


def build_gnn_fusion(**kwargs):
    return PerVisitGNNFusion( **kwargs)
