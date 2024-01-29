import torch
import typer
from halo import Halo
from termcolor import colored
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer
from typing_extensions import Annotated

from ragtime.device import get_device


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
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq",
        index_name="custom",
        passages_path="/mnt/disks/data/wiki_dpr",
        index_path="/mnt/disks/data/wiki_dpr.faiss",
    )
    dataset = retriever.index.dataset
    print(dataset)
    print("loading model...")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq",
        retriever=retriever,
    )
    print("preparing input...")
    input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")

    # returns RetrievAugLMOutput ?
    with Halo(text="generating...", spinner="dots") as spinner:
        result = model(**input_dict, output_retrieved=True)
        spinner.succeed()
    print(type(result))
    print(result.logits.shape)
    token_ids = torch.argmax(result.logits, dim=2)  # -1 dimension?
    print(token_ids.shape)
    print(token_ids)
    if citations:
        _print_docs(
            result.retrieved_doc_ids, result.doc_scores, dataset, print_passages=sources
        )
    print(tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0])


def _print_docs(doc_ids, doc_scores, dataset, print_passages=False):
    doc_ids = torch.flatten(doc_ids).tolist()
    docs = list(zip(doc_ids, torch.flatten(doc_scores).tolist()))
    docs.sort(key=lambda x: x[1], reverse=True)
    for id_, score in docs:
        print(f"{dataset[id_]['title']} ({score:.2f})", end="")
        if print_passages:
            passage = colored(dataset[id_]["text"], "dark_grey")
            print(passage, end="")
        print()


if __name__ == "__main__":
    typer.run(main)
