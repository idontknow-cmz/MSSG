"""Dataset-specific preprocessing dispatch for DLPFC and human breast cancer."""

from importlib import import_module

LOADERS = {"DLPFC": "dlpfc", "Human_Breast_Cancer": "human_breast_cancer"}


def prepare_data(dataset, data_path, **parameters):
    """Return processed AnnData using explicit arguments for the selected dataset loader."""
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset: {dataset}")
    module = import_module(f"{__package__}.{LOADERS[dataset]}")
    return module.load_ST_file(data_path, **parameters)
