"""Reconstruction, pair-mean spatial, and local contrastive objectives."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SpatialContrastiveLoss(nn.Module):
    """Bidirectional local contrast between positive and corrupted spot embeddings.

    Corrupt expression rows, encode them with the same weighted spatial graph,
    and fuse with the unchanged semantic embedding before calling this loss.
    The mask is the original binary spatial-neighbor mask, not weighted edges.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.discriminator = nn.Bilinear(feature_dim, feature_dim, 1)
        nn.init.xavier_uniform_(self.discriminator.weight)
        nn.init.zeros_(self.discriminator.bias)

    def forward(self, positive, corrupted, mask):
        if positive.shape != corrupted.shape or mask.shape != (
            len(positive),
            len(positive),
        ):
            raise ValueError(
                "Embeddings and neighbor mask must have matching spot dimensions"
            )
        degree = mask.sum(1, keepdim=True)
        if torch.any(degree <= 0):
            raise ValueError("Every contrastive readout requires at least one neighbor")
        positive, corrupted = F.relu(positive), F.relu(corrupted)
        summary = torch.sigmoid(F.normalize(torch.mm(mask, positive) / degree, dim=1))
        corrupted_summary = torch.sigmoid(
            F.normalize(torch.mm(mask, corrupted) / degree, dim=1)
        )
        logits = torch.cat(
            (
                self.discriminator(positive, summary),
                self.discriminator(corrupted, summary),
            ),
            dim=1,
        )
        inverse_logits = torch.cat(
            (
                self.discriminator(corrupted, corrupted_summary),
                self.discriminator(positive, corrupted_summary),
            ),
            dim=1,
        )
        target = torch.stack(
            (
                torch.ones(len(positive), device=positive.device),
                torch.zeros(len(positive), device=positive.device),
            ),
            dim=1,
        )
        return F.binary_cross_entropy_with_logits(
            logits, target
        ) + F.binary_cross_entropy_with_logits(inverse_logits, target)


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class NB(object):

    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1000000.0))
        t1 = (
            torch.lgamma(theta + self.eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + self.eps)
        )
        t2 = (theta + y_true) * torch.log(
            1.0 + y_pred / (theta + self.eps)
        ) + y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):

    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1000000.0))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(
            1.0 - self.pi + eps
        )
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + (1.0 - self.pi) * zero_nb + eps)
        result = torch.where(torch.lt(y_true, 1e-08), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


def spatial_pair_mean_loss(
    embedding: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    """Average positive and negative spatial log losses separately by their mask weights."""
    n = embedding.shape[0]
    if embedding.ndim != 2 or positive.shape != (n, n) or negative.shape != (n, n):
        raise ValueError("Spatial masks must match the embedding's spot count")
    for mask in (positive, negative):
        if (
            mask.device != embedding.device
            or not torch.isfinite(mask).all()
            or (mask < 0).any()
        ):
            raise ValueError(
                "Spatial masks must be finite/nonnegative and on the embedding device"
            )
    p_mass, n_mass = (positive.sum(), negative.sum())
    if p_mass <= 0 or n_mass <= 0:
        raise ValueError(
            "Pair-mean spatial loss requires both positive and negative mass"
        )
    probability = torch.sigmoid(cosine_similarity(embedding))
    return -0.5 * (
        (positive * torch.log(probability)).sum() / p_mass
        + (negative * torch.log(1 - probability)).sum() / n_mass
    )
