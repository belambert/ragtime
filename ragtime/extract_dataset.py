"""
Extract the wiki_dpr index and data into individual files/folders on disk
that we can point the training and inference scripts to.
"""

import typer
from transformers import RagRetriever


def main() -> None:
    """
    Save the wiki_dpr index and dataset to their own folders.
    """
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
    )
    dataset = retriever.index.dataset
    print("saving index...")
    dataset.get_index("embeddings").save("~/wiki_dpr_index")
    print("saving dataset...")
    dataset.drop_index("embeddings")
    dataset.save_to_disk("~/wiki_dpr")


def cli() -> None:
    """Run main() using typer."""
    typer.run(main)


if __name__ == "__main__":
    cli()
