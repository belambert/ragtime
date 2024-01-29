from contextlib import redirect_stderr
from pathlib import Path

import torch
import typer
from halo import Halo
from termcolor import colored
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer
from typing_extensions import Annotated

from ragtime.device import get_device

DATA_PREFIX = "/mnt/disks/data/"


def main(
    query: Annotated[str, typer.Argument()],
    citations: Annotated[bool, typer.Option()] = False,
    sources: Annotated[bool, typer.Option()] = False,
):
    """
    Do RAG inference on the given query.
    """
    device = get_device()
    print(device)

    print("loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained(
        "facebook/rag-token-nq",
    )
    print("loading retriever...")
    if Path(DATA_PREFIX).exists():
        with redirect_stderr(None):
            retriever = RagRetriever.from_pretrained(
                "facebook/rag-token-nq",
                index_name="custom",
                passages_path="/mnt/disks/data/wiki_dpr",
                index_path="/mnt/disks/data/wiki_dpr.faiss",
            )
    else:
        with redirect_stderr(None):
            retriever = RagRetriever.from_pretrained(
                "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
            )
    dataset = retriever.index.dataset
    print("loading model...")
    with redirect_stderr(None):
        model = RagTokenForGeneration.from_pretrained(
            "facebook/rag-token-nq",
            retriever=retriever,
        )
    input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")

    with Halo(text="generating...", spinner="dots") as spinner:
        result = model(**input_dict, output_retrieved=True)  # RetrievAugLMMarginOutput
        generated = model.generate(**input_dict)
        spinner.succeed()
    print("Answer:", end="")
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
    if citations:
        _print_docs(
            result.retrieved_doc_ids, result.doc_scores, dataset, print_passages=sources
        )


def _print_docs(doc_ids, doc_scores, dataset, print_passages=False):
    doc_ids = torch.flatten(doc_ids).tolist()
    docs = list(zip(doc_ids, torch.flatten(doc_scores).tolist()))
    docs.sort(key=lambda x: x[1], reverse=True)
    print(colored("Sources", attrs=["underline"]))
    for id_, score in docs:
        print(f"{dataset[id_]['title']} ({score:.2f})", end="")
        if print_passages:
            passage = " - " + colored(dataset[id_]["text"], "dark_grey")
            print(passage, end="")
        print()


if __name__ == "__main__":
    typer.run(main)
