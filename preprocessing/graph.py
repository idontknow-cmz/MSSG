"""Expression and spatial graph construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


def semantic_reweighted_spatial(raw_spatial, semantic_sources, *, strength):
    """Reweight existing spatial edges using equal-source semantic cosine ranks.

    Use the frozen source arrays before model-side centering. Edge support and
    the prepared diagonal are preserved. Add identity and row-normalize using
    the same convention as the prepared graph loader. No labels are used.
    """
    from scipy.stats import rankdata

    if not 0 <= strength < 1:
        raise ValueError("strength must be between zero (inclusive) and one")
    raw = sp.csr_matrix(raw_spatial, dtype=np.float32).copy()
    raw.sort_indices()
    if raw.shape[0] != raw.shape[1] or (raw != raw.T).nnz:
        raise ValueError("A symmetric square spatial graph is required")
    edges = sp.triu(raw, k=1, format="coo")
    if not edges.nnz or np.any(edges.data <= 0) or len(semantic_sources) != 3:
        raise ValueError(
            "Positive spatial edges and three semantic sources are required"
        )
    scores = np.zeros(edges.nnz, dtype=np.float32)
    for source in semantic_sources:
        source = np.asarray(source, dtype=np.float32)
        if (
            source.ndim != 2
            or source.shape[0] != raw.shape[0]
            or (not np.isfinite(source).all())
        ):
            raise ValueError(
                "Semantic arrays must be finite and aligned with graph rows"
            )
        norms = np.linalg.norm(source, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Semantic source contains a zero vector")
        unit = source / norms
        for start in range(0, edges.nnz, 512):
            stop = start + 512
            left, right = (edges.row[start:stop], edges.col[start:stop])
            scores[start:stop] += np.einsum("ij,ij->i", unit[left], unit[right]) / 3.0
    ranks = (
        np.zeros_like(scores)
        if edges.nnz == 1 or np.ptp(scores) == 0
        else (
            2 * (rankdata(scores, method="average") - 1) / (edges.nnz - 1) - 1
        ).astype(np.float32)
    )
    upper = sp.coo_matrix(
        (edges.data * (1 + strength * ranks), (edges.row, edges.col)), shape=raw.shape
    ).tocsr()
    weighted = upper + upper.T + sp.diags(raw.diagonal(), format="csr")
    with_identity = weighted + sp.eye(raw.shape[0])
    degree = np.asarray(with_identity.sum(1))
    inverse_degree = np.power(degree, -1).flatten()
    inverse_degree[np.isinf(inverse_degree)] = 0
    normalized = sp.diags(inverse_degree).dot(with_identity).astype(np.float32)
    return torch.from_numpy(normalized.toarray())


def spatial_construct_graph1(adata, radius):
    coor = pd.DataFrame(adata.obsm["spatial"])
    coor.index = adata.obs.index
    coor.columns = ["imagerow", "imagecol"]
    A = np.zeros((coor.shape[0], coor.shape[0]))
    nbrs = NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]] = 1
    print("The graph contains %d edges, %d cells." % (sum(sum(A)), adata.n_obs))
    print("%.4f neighbors per cell on average." % (sum(sum(A)) / adata.n_obs))
    graph_nei = torch.from_numpy(A)
    graph_neg = torch.ones(coor.shape[0], coor.shape[0]) - graph_nei
    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return (sadj, graph_nei, graph_neg)


def spatial_construct_graph(positions, k):
    """Build a binary spatial graph by searching a distance threshold."""
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 3:
        raise ValueError("Expected at least three spots with two spatial coordinates")
    if not np.isfinite(positions).all():
        raise ValueError("Spatial coordinates must be finite")
    if (
        isinstance(k, bool)
        or not isinstance(k, (int, np.integer))
        or not 0 < k < len(positions)
    ):
        raise ValueError(
            "k must be an integer between one and the number of spots minus one"
        )
    A = euclidean_distances(positions)
    tmp = 0
    mink = 2
    for t in range(100, 1000, 100):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 100, 1000, 10):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    found = False
    for t in range(tmp - 10, 1000, 5):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            A = A1
            found = True
            break
    if not found:
        raise ValueError(
            "No valid spatial distance threshold was found. Check coordinate units "
            "and k; no adjacency graph was produced."
        )
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    graph_nei = torch.from_numpy(A)
    graph_neg = torch.ones(positions.shape[0], positions.shape[0]) - graph_nei
    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return (sadj, graph_nei, graph_neg)
