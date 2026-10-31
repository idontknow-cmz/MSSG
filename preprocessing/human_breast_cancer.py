"""Human breast cancer data loading, preprocessing, and graph construction."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from .graph import spatial_construct_graph


def normalize(adata, highly_genes):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(path, highly_genes, k1):
    labels_path = os.path.join(path, "metadata.tsv")
    metadata = pd.read_table(labels_path, sep="\t", index_col=0)
    adata = sc.read_visium(
        path, count_file="filtered_feature_bc_matrix.h5", load_images=True
    )
    adata.var_names_make_unique()
    if metadata.index.has_duplicates or not adata.obs_names.isin(metadata.index).all():
        raise ValueError(
            "Metadata must contain a unique barcode for every expression spot"
        )
    labels = metadata["ground_truth"].reindex(adata.obs_names)
    if labels.isna().any():
        raise ValueError("HBC metadata contains missing ground-truth labels")
    _, ground = np.unique(labels, return_inverse=True)
    cell_labels = labels.str[0].map({"D": 1, "H": 0, "I": 1, "T": 1})
    if cell_labels.isna().any():
        raise ValueError("HBC annotation labels must start with D, H, I, or T")
    adata.obs["ground_truth"] = labels
    adata.obs["ground"] = ground
    adata.obs["annot_type"] = cell_labels.to_numpy(dtype=int)
    adata.var_names_make_unique()
    adata.X = np.array(sp.csr_matrix(adata.X, dtype=np.float32).todense())
    print(adata)
    adata = normalize(adata, highly_genes=highly_genes)
    sadj, graph_nei, graph_neg = spatial_construct_graph(adata.obsm["spatial"], k=k1)
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata
