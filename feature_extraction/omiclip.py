"""OmiCLIP text-tower loading and gene-sequence feature extraction."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from open_clip import get_tokenizer
from open_clip.transformer import TextTransformer
from tqdm import tqdm

OFFICIAL_REPOSITORY = "https://github.com/GuangyuWangLab2021/Loki"
OFFICIAL_COMMIT = "c6d3eed94dc18c5c19a10120828578d5a09d5347"
OFFICIAL_MODEL = "WangGuangyuLab/Loki"
MODEL_NAME = "coca_ViT-L-14"
TEXT_CONFIG = {
    "context_length": 76,
    "vocab_size": 49408,
    "width": 768,
    "heads": 12,
    "layers": 12,
    "output_dim": 768,
    "embed_cls": True,
    "output_tokens": True,
}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested, but CUDA is unavailable")
    return torch.device(requested)


def build_text_model() -> TextTransformer:
    return TextTransformer(**TEXT_CONFIG)


def extract_text_tower(
    checkpoint_path: Path, output_path: Path, force: bool = False
) -> None:
    if output_path.is_file() and (not force):
        print(f"Text-only checkpoint already exists: {output_path}")
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Official checkpoint not found: {checkpoint_path}")
    print(f"Memory-mapping trusted official checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    if "state_dict" not in checkpoint:
        raise KeyError("Official checkpoint has no 'state_dict' entry")
    full_state = checkpoint["state_dict"]
    text_state = {
        key.removeprefix("text."): value.detach().cpu().clone()
        for key, value in full_state.items()
        if key.startswith("text.")
    }
    if len(text_state) != 150:
        raise ValueError(f"Expected 150 text tensors, found {len(text_state)}")
    model = build_text_model()
    model.load_state_dict(text_state, strict=True)
    tensor_bytes = sum(
        (value.numel() * value.element_size() for value in text_state.values())
    )
    payload = {
        "state_dict": text_state,
        "text_config": TEXT_CONFIG,
        "source": {
            "repository": OFFICIAL_REPOSITORY,
            "repository_commit": OFFICIAL_COMMIT,
            "model_repository": OFFICIAL_MODEL,
            "full_checkpoint_name": checkpoint_path.name,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_run_name": checkpoint.get("name"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".incomplete")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Extracted text tensors: {len(text_state)}")
    print(f"Text tensor bytes: {tensor_bytes} ({tensor_bytes / 2 ** 30:.3f} GiB)")
    print(f"Text-only checkpoint saved: {output_path}")
    del model, payload, text_state, full_state, checkpoint
    gc.collect()


def load_text_tower(path: Path, device: torch.device) -> TextTransformer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("text_config")
    if config != TEXT_CONFIG:
        raise ValueError(f"Unexpected text-tower configuration: {config}")
    model = build_text_model()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    return model


def load_housekeeping_genes(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"Housekeeping CSV not found: {path}")
    frame = pd.read_csv(path)
    if "genesymbol" in frame.columns:
        values: Iterable[object] = frame["genesymbol"]
    elif len(frame.columns) == 1:
        values = frame.iloc[:, 0]
    else:
        raise ValueError(
            "Housekeeping CSV must contain a 'genesymbol' column or exactly one column"
        )
    return {str(value) for value in values if pd.notna(value)}


def build_gene_sentences(adata, top_k: int, housekeeping_genes: set[str]) -> list[str]:
    gene_names = np.asarray(adata.var_names.astype(str))
    allowed = np.array(
        [
            "." not in gene and "-" not in gene and (gene not in housekeeping_genes)
            for gene in gene_names
        ],
        dtype=bool,
    )
    gene_names = gene_names[allowed]
    matrix = adata.X[:, allowed]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    sentences: list[str] = []
    for row_index in tqdm(range(matrix.shape[0]), desc="Building gene sentences"):
        start, end = matrix.indptr[row_index : row_index + 2]
        columns = matrix.indices[start:end]
        values = matrix.data[start:end]
        order = np.argsort(-values, kind="stable")[:top_k]
        sentences.append(" ".join(gene_names[columns[order]]))
    if any((not sentence for sentence in sentences)):
        raise ValueError("At least one spot produced an empty gene sentence")
    return sentences


@torch.inference_mode()
def encode_sentences(
    model: TextTransformer, sentences: list[str], batch_size: int, device: torch.device
) -> np.ndarray:
    tokenizer = get_tokenizer(MODEL_NAME)
    batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(sentences), batch_size), desc="Encoding OmiCLIP"):
        batch = sentences[start : start + batch_size]
        tokens = tokenizer(batch).to(device)
        output = model(tokens)
        pooled = output[0] if isinstance(output, tuple) else output
        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        batches.append(pooled.cpu().numpy())
    embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if embeddings.shape != (len(sentences), 768):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    return embeddings
