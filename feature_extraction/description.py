"""Description features encoded from ranked highly expressed genes."""

from .common import build_spot_texts, generate_embeddings, ranked_top_genes


def extract_description(
    adata, tokenizer, model, *, n_top_genes, batch_size, max_length
):
    genes = ranked_top_genes(adata, n_top_genes)
    texts = build_spot_texts(genes, "description", None, 0, 0)
    return generate_embeddings(
        tokenizer, model, texts, "description", batch_size, max_length
    )
