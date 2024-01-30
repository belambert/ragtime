from contextlib import redirect_stderr
from pathlib import Path

from transformers import RagModel, RagRetriever, RagTokenForGeneration, RagTokenizer

DATA_FOLDER = Path("/mnt/disks/data/")


def load_model(debug: bool = False) -> tuple[RagTokenizer, RagModel]:
    """Load the RAG model. If debug=True, use the dummy dataset."""
    print("loading tokenizer...")
    with redirect_stderr(None):
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

    print("loading retriever...")
    if debug:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        )
    elif DATA_FOLDER.exists():
        with redirect_stderr(None):
            retriever = RagRetriever.from_pretrained(
                "facebook/rag-token-nq",
                index_name="custom",
                passages_path=DATA_FOLDER / "/wiki_dpr",
                index_path=DATA_FOLDER / "wiki_dpr.faiss",
            )
    else:
        with redirect_stderr(None):
            retriever = RagRetriever.from_pretrained(
                "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
            )
    print("loading model...")
    with redirect_stderr(None):
        model = RagTokenForGeneration.from_pretrained(
            "facebook/rag-token-nq", retriever=retriever
        )
    return tokenizer, model
