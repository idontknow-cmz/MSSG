"""Shared annotation, text construction, and language-model encoding utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

PROMPT_CONFIGS = {
    "description": {
        "system": "You are an expert in bioinformatics. Given the following ranked highly expressed genes in a spatial transcriptomics spot, represent its tissue or cell-type context, disease or phenotype associations, and protein-level relationships that are not merely a restatement of pathway or Gene Ontology membership. Produce a dense representation of this description-level state.",
        "user": "Ranked highly expressed genes: ",
    },
    "pathway": {
        "system": "You are an expert in bioinformatics. Represent the pathway-level state of a spatial transcriptomics spot using only the supplied pathway and Gene Ontology terms. Preserve the association between each gene and its listed terms, and do not add biological interpretations or annotations that are not supplied.",
        "user": "Genes and their supplied GO/pathway annotations: ",
    },
}


def ranked_top_genes(adata, n_top_genes: int) -> list[list[str]]:
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be at least 1")
    n_top_genes = min(n_top_genes, adata.n_vars)
    matrix = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    indices = np.argsort(matrix, axis=1)[:, -n_top_genes:][:, ::-1]
    names = np.asarray(adata.var_names.astype(str))
    return [names[row].tolist() for row in indices]


def _as_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item


def _clean_term(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    term = " ".join(value.split()).strip(" ;,")
    return term or None


def extract_annotation_terms(hit: dict[str, Any]) -> list[str]:
    """Flatten MyGene GO/pathway structures into deterministic readable terms."""
    terms: list[str] = []
    go = hit.get("go")
    if isinstance(go, dict):
        for namespace in ("BP", "MF", "CC"):
            for record in _as_records(go.get(namespace)):
                term = _clean_term(record.get("term"))
                if term:
                    terms.append(f"GO-{namespace}: {term}")
    pathway = hit.get("pathway")
    if isinstance(pathway, dict):
        for database in sorted(pathway, key=str.casefold):
            for record in _as_records(pathway[database]):
                term = _clean_term(record.get("name") or record.get("term"))
                if term:
                    terms.append(f"{database}: {term}")
    return list(dict.fromkeys(terms))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def query_mygene_annotations(
    genes: Iterable[str], species: str, cache_path: Path
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Query missing symbols and merge results into a reusable species cache."""
    try:
        import mygene
    except ImportError as exc:
        raise RuntimeError(
            "Pathway annotation requires mygene. Install it with: pip install mygene"
        ) from exc
    requested = list(
        dict.fromkeys((str(gene).strip() for gene in genes if str(gene).strip()))
    )
    cache = load_json(cache_path)
    missing = [gene for gene in requested if gene not in cache]
    if missing:
        client = mygene.MyGeneInfo()
        for start in tqdm(
            range(0, len(missing), 1000), desc=f"Querying MyGene ({species})"
        ):
            batch = missing[start : start + 1000]
            hits = client.querymany(
                batch,
                scopes="symbol",
                fields="symbol,go,pathway",
                species=species,
                as_dataframe=False,
                returnall=False,
                verbose=False,
            )
            by_query: dict[str, dict[str, Any]] = {}
            for hit in hits:
                query = str(hit.get("query", ""))
                if query and query not in by_query and (not hit.get("notfound", False)):
                    by_query[query] = hit
            for gene in batch:
                cache[gene] = extract_annotation_terms(by_query.get(gene, {}))
        save_json(cache_path, cache)
    mapping = {gene: list(cache.get(gene, [])) for gene in requested}
    annotated = sum((bool(mapping[gene]) for gene in requested))
    report = {
        "species": species,
        "requested_unique_genes": len(requested),
        "annotated_unique_genes": annotated,
        "coverage": annotated / len(requested) if requested else 0.0,
        "newly_queried_genes": len(missing),
        "cache_file": str(cache_path.resolve()),
    }
    return (mapping, report)


def build_spot_texts(
    ranked_genes: list[list[str]],
    source: str,
    gene2pathway: dict[str, list[str]] | None,
    max_terms_per_gene: int,
    max_terms_per_spot: int,
) -> list[str]:
    if source == "description":
        return [", ".join(genes) for genes in ranked_genes]
    if source != "pathway":
        raise ValueError(f"Unsupported semantic source: {source}")
    if gene2pathway is None:
        raise ValueError("gene2pathway is required for the pathway source")
    texts: list[str] = []
    for genes in ranked_genes:
        remaining = max_terms_per_spot
        entries: list[str] = []
        for gene in genes:
            selected = gene2pathway.get(gene, [])[: min(max_terms_per_gene, remaining)]
            if selected:
                entries.append(f"{gene}: {', '.join(selected)}")
                remaining -= len(selected)
            if remaining <= 0:
                break
        if not entries:
            entries.append("No supplied GO/pathway terms for " + ", ".join(genes))
        texts.append("; ".join(entries))
    return texts


def load_embedding_model(model_path: str, device: str, load_in_4bit: bool):
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    local_only = Path(model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=local_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"trust_remote_code": True, "local_files_only": local_only}
    if load_in_4bit:
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("4-bit generation requires an available CUDA device")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device}
    else:
        kwargs["torch_dtype"] = (
            torch.float16 if device.startswith("cuda") else torch.float32
        )
    model = AutoModel.from_pretrained(model_path, **kwargs)
    if not load_in_4bit:
        model.to(device)
    model.requires_grad_(False)
    model.eval()
    return (tokenizer, model)


def generate_embeddings(
    tokenizer, model, texts: list[str], source: str, batch_size: int, max_length: int
) -> np.ndarray:
    prompt = PROMPT_CONFIGS[source]
    output_batches: list[np.ndarray] = []
    model_device = next(model.parameters()).device
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {source}"):
        prompts = [
            f"<|im_start|>system\n{prompt['system']}<|im_end|>\n"
            f"<|im_start|>user\n{prompt['user']}{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for text in texts[start : start + batch_size]
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model_device)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=False, return_dict=True)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        output_batches.append(pooled.float().cpu().numpy())
    result = np.concatenate(output_batches, axis=0).astype(np.float32, copy=False)
    if result.shape[0] != len(texts):
        raise RuntimeError(f"Embedding row mismatch: {result.shape[0]} != {len(texts)}")
    if not np.isfinite(result).all():
        raise RuntimeError("Generated embeddings contain NaN or infinity")
    return result


def validate_existing(path: Path, n_obs: int) -> tuple[int, int]:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[0] != n_obs:
        raise RuntimeError(f"Invalid existing embedding shape at {path}: {array.shape}")
    if array.dtype != np.float32:
        raise RuntimeError(f"Invalid existing embedding dtype at {path}: {array.dtype}")
    return (int(array.shape[0]), int(array.shape[1]))
