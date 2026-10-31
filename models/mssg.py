"""MSSG with semantic-reweighted spatial encoding and gated low-rank fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .fusion import GatedLowRankResidualFusion, SemanticFusionBottleneck
from .layers import GCN, LLMHead, ZINBDecoder


@dataclass
class MSSGOutput:
    """Latent representation, reconstruction parameters, and intermediate features."""

    latent: torch.Tensor
    pi: torch.Tensor
    disp: torch.Tensor
    mean: torch.Tensor
    expression_embedding: torch.Tensor
    semantic_heads: list[torch.Tensor]
    semantic_embedding: torch.Tensor


class MSSG(nn.Module):
    """Single spatial GCN with frozen semantic heads and bottleneck fusion."""

    def __init__(
        self,
        *,
        nfeat: int,
        hidden_dim: int,
        latent_dim: int,
        expression_dim: int,
        semantic_dims: Sequence[int],
        bottleneck_dim: int,
        bottleneck_layers: int,
        attention_heads: int,
        rank: int,
        dropout: float,
        semantic_skip_weight: float,
        gate_initial_probability: float,
    ) -> None:
        super().__init__()
        self.semantic_dims = tuple(int(dim) for dim in semantic_dims)
        if len(self.semantic_dims) != 3 or any(dim <= 0 for dim in self.semantic_dims):
            raise ValueError(
                "semantic_dims must contain three positive source dimensions"
            )
        if min(nfeat, hidden_dim, latent_dim, expression_dim, bottleneck_dim) <= 0:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between zero and one")
        if not math.isfinite(semantic_skip_weight) or semantic_skip_weight < 0:
            raise ValueError("semantic_skip_weight must be finite and nonnegative")
        self.semantic_skip_weight = float(semantic_skip_weight)
        self.spatial_encoder = GCN(
            nfeat, hidden_dim, expression_dim, dropout, n_layers=2
        )
        self.decoder = ZINBDecoder(nfeat, hidden_dim, latent_dim)
        self.semantic_heads = nn.ModuleList(
            LLMHead(width, latent_dim) for width in self.semantic_dims
        )
        self.semantic_heads.requires_grad_(False)
        self.semantic_fusion = SemanticFusionBottleneck(
            dims=[latent_dim] * 3,
            out_dim=latent_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=bottleneck_layers,
            n_heads=attention_heads,
            dropout=dropout,
        )
        self.semantic_expression_fusion = GatedLowRankResidualFusion(
            dim_sem=latent_dim,
            dim_expr=expression_dim,
            out_dim=latent_dim,
            rank=rank,
            post_hidden_dim=hidden_dim,
            dropout=dropout,
            gate_initial_probability=gate_initial_probability,
        )

    def encode_semantics(
        self, inputs: Sequence[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Center and unit-normalize each source, then fuse its frozen projection."""
        if len(inputs) != 3:
            raise ValueError("Expected Description, Pathway, and OmiCLIP features")
        n_spots = inputs[0].shape[0]
        for index, (features, width) in enumerate(zip(inputs, self.semantic_dims)):
            if features.shape != (n_spots, width):
                raise ValueError(
                    f"Semantic source {index} must have shape {(n_spots, width)}"
                )
        centered = [
            F.normalize(features - features.mean(0, keepdim=True), dim=1)
            for features in inputs
        ]
        heads = [
            head(features) for head, features in zip(self.semantic_heads, centered)
        ]
        semantic = self.semantic_fusion(heads)
        residual = [
            F.normalize(head - head.mean(0, keepdim=True), dim=1, eps=1e-6)
            * math.sqrt(head.shape[1])
            for head in heads
        ]
        semantic = semantic + self.semantic_skip_weight * torch.stack(residual).mean(0)
        return heads, semantic

    def forward(
        self,
        x: torch.Tensor,
        weighted_spatial: torch.Tensor,
        semantic_inputs: Sequence[torch.Tensor],
    ) -> MSSGOutput:
        """Encode spots using the fixed, normalized semantic-reweighted spatial graph.

        Build weighted_spatial once with preprocessing.graph.semantic_reweighted_spatial.
        All semantic inputs must have the same spot ordering as x. Semantic centering
        uses the supplied full set of spots, matching the full-batch model.
        """
        if weighted_spatial.shape != (x.shape[0], x.shape[0]):
            raise ValueError("Spatial adjacency must match the expression spot count")
        if any(features.shape[0] != x.shape[0] for features in semantic_inputs):
            raise ValueError("Semantic inputs must match the expression spot count")
        expression = self.spatial_encoder(x, weighted_spatial)
        heads, semantic = self.encode_semantics(semantic_inputs)
        latent = self.semantic_expression_fusion(semantic, expression)
        pi, disp, mean = self.decoder(latent)
        return MSSGOutput(latent, pi, disp, mean, expression, heads, semantic)


__all__ = ["MSSG", "MSSGOutput"]
