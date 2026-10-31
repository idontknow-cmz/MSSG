"""Semantic-source fusion and semantic-expression fusion modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _validate_dims(dims: Sequence[int], out_dim: int) -> tuple[int, ...]:
    dims = tuple((int(dim) for dim in dims))
    if not dims or any((dim <= 0 for dim in dims)):
        raise ValueError(f"dims must contain positive dimensions, got {dims}")
    if out_dim <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}")
    return dims


def _validate_semantic_inputs(
    inputs: Sequence[torch.Tensor], dims: tuple[int, ...]
) -> int:
    if len(inputs) != len(dims):
        raise ValueError(
            f"Expected {len(dims)} semantic sources, received {len(inputs)}"
        )
    n_spots: int | None = None
    for index, (tensor, expected_dim) in enumerate(zip(inputs, dims)):
        if tensor.ndim != 2:
            raise ValueError(
                f"Source {index} must be 2D, got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != expected_dim:
            raise ValueError(
                f"Source {index} feature mismatch: expected {expected_dim}, got {tensor.shape[1]}"
            )
        if n_spots is None:
            n_spots = tensor.shape[0]
        elif tensor.shape[0] != n_spots:
            raise ValueError(
                f"All sources must have the same spot count; source {index} has {tensor.shape[0]}, expected {n_spots}"
            )
    assert n_spots is not None
    return n_spots


def _validate_attention_dim(embed_dim: int, n_heads: int) -> None:
    if embed_dim <= 0 or n_heads <= 0:
        raise ValueError("embed_dim and n_heads must be positive")
    if embed_dim % n_heads != 0:
        raise ValueError(
            f"Attention dimension {embed_dim} must be divisible by n_heads={n_heads}"
        )


class _BottleneckExchangeLayer(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.bottleneck_reads_sources = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.sources_read_bottleneck = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.bottleneck_norm = nn.LayerNorm(embed_dim)
        self.source_norm = nn.LayerNorm(embed_dim)

    def forward(self, source_tokens: torch.Tensor, bottleneck_tokens: torch.Tensor):
        update, _ = self.bottleneck_reads_sources(
            bottleneck_tokens, source_tokens, source_tokens, need_weights=False
        )
        bottleneck_tokens = self.bottleneck_norm(bottleneck_tokens + update)
        update, _ = self.sources_read_bottleneck(
            source_tokens, bottleneck_tokens, bottleneck_tokens, need_weights=False
        )
        source_tokens = self.source_norm(source_tokens + update)
        return (source_tokens, bottleneck_tokens)


class SemanticFusionBottleneck(nn.Module):
    """Exchange semantic-source information through learnable bottleneck tokens.

    Token initialization parameters are shared across spots. Attention updates
    are computed independently within each spot."""

    def __init__(
        self,
        dims: Sequence[int],
        out_dim: int,
        bottleneck_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dims = _validate_dims(dims, out_dim)
        self.out_dim = int(out_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        _validate_attention_dim(self.bottleneck_dim, n_heads)
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.source_projections = nn.ModuleList(
            (nn.Linear(dim, self.bottleneck_dim) for dim in self.dims)
        )
        self.bottleneck_tokens = nn.Parameter(torch.empty(1, 1, self.bottleneck_dim))
        nn.init.normal_(self.bottleneck_tokens, std=0.02)
        self.layers = nn.ModuleList(
            (
                _BottleneckExchangeLayer(self.bottleneck_dim, n_heads, dropout)
                for _ in range(n_layers)
            )
        )
        self.output = nn.Linear(len(self.dims) * self.bottleneck_dim, self.out_dim)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        n_spots = _validate_semantic_inputs(inputs, self.dims)
        source_tokens = torch.stack(
            [
                projection(features)
                for projection, features in zip(self.source_projections, inputs)
            ],
            dim=1,
        )
        bottleneck_tokens = self.bottleneck_tokens.expand(n_spots, -1, -1)
        for layer in self.layers:
            source_tokens, bottleneck_tokens = layer(source_tokens, bottleneck_tokens)
        return self.output(source_tokens.flatten(start_dim=1))


def _validate_dimensions(dim_sem: int, dim_expr: int, out_dim: int) -> None:
    if min(dim_sem, dim_expr, out_dim) <= 0:
        raise ValueError(
            f"dim_sem, dim_expr, and out_dim must be positive; got {dim_sem}, {dim_expr}, {out_dim}"
        )


def _validate_modality_inputs(
    z_sem: torch.Tensor, z_expr: torch.Tensor, dim_sem: int, dim_expr: int
) -> int:
    if z_sem.ndim != 2 or z_expr.ndim != 2:
        raise ValueError(
            f"Expected two 2D tensors, got {tuple(z_sem.shape)} and {tuple(z_expr.shape)}"
        )
    if z_sem.shape[0] != z_expr.shape[0]:
        raise ValueError(
            f"Spot count mismatch: semantic={z_sem.shape[0]}, expression={z_expr.shape[0]}"
        )
    if z_sem.shape[1] != dim_sem or z_expr.shape[1] != dim_expr:
        raise ValueError(
            f"Feature mismatch: expected semantic/expression={dim_sem}/{dim_expr}, got {z_sem.shape[1]}/{z_expr.shape[1]}"
        )
    if z_sem.device != z_expr.device:
        raise ValueError(f"Device mismatch: {z_sem.device} vs {z_expr.device}")
    if z_sem.dtype != z_expr.dtype:
        raise ValueError(f"dtype mismatch: {z_sem.dtype} vs {z_expr.dtype}")
    return int(z_sem.shape[0])


class GatedLowRankResidualFusion(nn.Module):
    """Fuse expression and semantic features through a gated low-rank residual."""

    def __init__(
        self,
        dim_sem,
        dim_expr,
        out_dim,
        rank,
        post_hidden_dim,
        dropout,
        gate_initial_probability,
    ):
        super().__init__()
        _validate_dimensions(dim_sem, dim_expr, out_dim)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        self.dim_sem, self.dim_expr, self.rank = (int(dim_sem), int(dim_expr), rank)
        hidden = int(post_hidden_dim or dim_expr)
        if hidden <= 0:
            raise ValueError("post_hidden_dim must be positive")
        self.expr_factor = nn.Linear(dim_expr, rank, bias=False)
        self.sem_factor = nn.Linear(dim_sem, rank, bias=False)
        self.interaction = nn.Linear(rank, dim_expr, bias=False)
        self.output_mlp = nn.Sequential(
            nn.Linear(dim_expr, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        if not 0 < gate_initial_probability < 1:
            raise ValueError("gate_initial_probability must be between zero and one")
        self.gate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(gate_initial_probability)))
        )

    def forward(self, z_sem, z_expr):
        _validate_modality_inputs(z_sem, z_expr, self.dim_sem, self.dim_expr)
        cross = self.interaction(self.expr_factor(z_expr) * self.sem_factor(z_sem))
        return self.output_mlp(z_expr + self.gate_logit.sigmoid() * cross)
