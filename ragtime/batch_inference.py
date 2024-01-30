"""
This script is a work in progress.
"""

import typer

# from transformers import RagTokenizer
from typing_extensions import Annotated

from ragtime.device import get_device
from ragtime.train import (
    get_preproc_function,
    load_model,
    load_ms_marco,
    pad_batch,
    print_batch,
)


def main(
    debug: Annotated[bool, typer.Option(help="use data subset")] = False,
    batch_size: Annotated[int, typer.Option(help="batch size")] = 4,
) -> None:
    """..."""

    device = get_device()
    print(device)
    tokenizer, model = load_model(False)
    # tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    dataset = load_ms_marco(debug)

    tok_data = dataset.map(
        get_preproc_function(tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    )
    tok_data.set_format("torch")
    model.eval()
    for partition in tok_data.keys():
        print(f"Partition: {partition}")
        data_iter = tok_data[partition].iter(batch_size)
        for _, batch in enumerate(data_iter):
            pad_batch(batch)
            output = model(**batch)
            print_batch(tokenizer, batch, output)
            # print it somewhere...


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
