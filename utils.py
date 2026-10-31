"""Sparse matrix, embedding loading, and label alignment utilities."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_sparse_matrix(mx):
    """Row-normalize a sparse matrix, leaving zero-sum rows as zero."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_embeddings(npy_path):
    """Load spot embeddings and return their tensor and feature dimension."""
    final_base_embeddings = np.load(npy_path)
    final_base_embeddings = torch.tensor(final_base_embeddings)
    llm_hidden_dim = final_base_embeddings.shape[-1]
    return (final_base_embeddings, llm_hidden_dim)


def hungarian_match(y_true, y_pred):
    """Align predicted cluster labels to reference labels using the Hungarian algorithm."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        cost_matrix[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([mapping[y] for y in y_pred])
    return y_pred_aligned
