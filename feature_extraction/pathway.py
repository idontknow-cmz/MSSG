"""Pathway features encoded from supplied gene-level GO and pathway annotations."""

from .common import build_spot_texts, generate_embeddings, ranked_top_genes


def extract_pathway(
    adata,
    tokenizer,
    model,
    gene2pathway,
    *,
    n_top_genes,
    max_terms_per_gene,
    max_terms_per_spot,
    batch_size,
    max_length,
):
    genes = ranked_top_genes(adata, n_top_genes)
    texts = build_spot_texts(
        genes, "pathway", gene2pathway, max_terms_per_gene, max_terms_per_spot
    )
    return generate_embeddings(
        tokenizer, model, texts, "pathway", batch_size, max_length
    )
