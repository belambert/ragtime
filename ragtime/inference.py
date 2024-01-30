"""
CLI to use a RAG model for inference.
"""

import torch
import typer
from datasets import Dataset
from halo import Halo
from termcolor import colored
from torch import Tensor
from typing_extensions import Annotated

from .device import get_device
from .model import load_model


def main(
    query: Annotated[str, typer.Argument()],
    articles: Annotated[bool, typer.Option(help="print names of articles")] = False,
    passages: Annotated[bool, typer.Option(help="print passages used")] = False,
) -> None:
    """
    Do RAG inference on the given query.
    """
    device = get_device()
    print(device)

    tokenizer, model = load_model(False)
    dataset = model.retriever.index.dataset

    input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")

    with Halo(text="generating...", spinner="dots") as spinner:
        result = model(**input_dict, output_retrieved=True)  # RetrievAugLMMarginOutput
        generated = model.generate(**input_dict)
        spinner.succeed()
    print("Answer:", end="")
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
    if articles:
        _print_docs(
            result.retrieved_doc_ids,
            result.doc_scores,
            dataset,
            print_passages=passages,
        )


def _print_docs(
    doc_ids: Tensor, doc_scores: Tensor, dataset: Dataset, print_passages: bool = False
) -> None:
    doc_ids_list = torch.flatten(doc_ids).tolist()
    docs = list(zip(doc_ids_list, torch.flatten(doc_scores).tolist()))
    docs.sort(key=lambda x: x[1], reverse=True)
    print(colored("Sources", attrs=["underline"]))
    for id_, score in docs:
        print(f"{dataset[id_]['title']} ({score:.2f})", end="")
        if print_passages:
            passage = " - " + colored(dataset[id_]["text"], "dark_grey")
            print(passage, end="")
        print()


def cli() -> None:
    """Run main() with typer."""
    typer.run(main)


if __name__ == "__main__":
    cli()
