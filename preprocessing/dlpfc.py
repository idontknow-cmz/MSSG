"""DLPFC data loading, expression preprocessing, and graph construction."""

from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .graph import spatial_construct_graph1


def normalize(adata, highly_genes):
    print("start select HVGs")
    library_sizes = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.layers["counts"] = adata.X.copy()
    median_library_size = np.median(library_sizes[library_sizes > 0])
    adata.obs["library_size"] = library_sizes
    adata.obs["size_factor"] = library_sizes / median_library_size
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(path, highly_genes, radius):
    labels_path = os.path.join(path, "metadata.tsv")
    metadata = pd.read_table(labels_path, sep="\t", index_col=0)
    adata1 = sc.read_visium(
        path, count_file="filtered_feature_bc_matrix.h5", load_images=True
    )
    adata1.var_names_make_unique()
    if metadata.index.has_duplicates or not adata1.obs_names.isin(metadata.index).all():
        raise ValueError(
            "Metadata must contain a unique barcode for every expression spot"
        )
    labels = metadata["layer_guess_reordered"].reindex(adata1.obs_names)
    NA_labels = np.where(labels.isnull())
    labels = labels.dropna()
    if labels.empty:
        raise ValueError("DLPFC metadata contains no labeled spots")
    _, ground = np.unique(labels, return_inverse=True)
    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm["spatial"]
    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)
    adata = ad.AnnData(
        pd.DataFrame(
            data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32
        )
    )
    adata.var_names_make_unique()
    adata.obs["ground_truth"] = labels
    adata.obs["ground"] = ground
    adata.obsm["spatial"] = positions
    adata.obs["array_row"] = adata1.obs["array_row"]
    adata.obs["array_col"] = adata1.obs["array_col"]
    adata.uns["spatial"] = adata1.uns["spatial"]
    adata.var["gene_ids"] = adata1.var["gene_ids"]
    adata.var["feature_types"] = adata1.var["feature_types"]
    adata.var["genome"] = adata1.var["genome"]
    adata.var_names_make_unique()
    adata = normalize(adata, highly_genes=highly_genes)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata
