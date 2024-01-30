import math
from itertools import chain
from typing import Callable

import torch
import typer
import wandb as wb
from datasets import Dataset, load_dataset
from torch import Tensor
from transformers import (
    BatchEncoding,
    RagModel,
    RagRetriever,
    RagTokenForGeneration,
    RagTokenizer,
)
from transformers.modeling_outputs import ModelOutput
from typing_extensions import Annotated

from ragtime.device import get_device

# Refer to:
# https://huggingface.co/datasets/wiki_dpr
# https://huggingface.co/facebook/rag-token-nq
# https://huggingface.co/docs/transformers/model_doc/rag

# For Ray finetuning see:
# https://huggingface.co/blog/ray-rag
# https://shamanesiri.medium.com/how-to-finetune-the-entire-rag-architecture-including-dpr-retriever-4b4385322552
# https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/finetune_rag.py


# RetrievAugLMOutput vs RetrievAugLMMarginOutput

# what to do about RagConfig? it has an option: output_retrieved

MAX_LENGTH = 128


# pylint: disable-next=too-many-locals
def main(
    debug: Annotated[bool, typer.Option(help="use data subset")] = False,
    epochs: Annotated[int, typer.Option(help="num epochs")] = 100,
    lr: Annotated[float, typer.Option(help="learning rate")] = 0.001,
    batch_size: Annotated[int, typer.Option(help="batch size")] = 4,
    wandb: Annotated[bool, typer.Option(help="use wandb")] = False,
) -> None:
    """Fine-tune a RAG model with the MS MARCO dataset."""
    run = wb.init(project="ragtime", mode="online" if wandb else "disabled")
    wb.config.learning_rate = lr
    wb.config.batch_size = batch_size

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
    params = chain(model.generator.parameters(), model.question_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    batches_per_epoch = math.ceil(len(tok_data["train"]) / batch_size)
    print(f"batches per epoch: {batches_per_epoch}")
    print("begin training loop...")
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = 0
        model.train()
        data_iter = tok_data["train"].iter(batch_size, drop_last_batch=True)
        for i, batch in enumerate(data_iter):
            print(f"batch: {i} / {batches_per_epoch}")
            pad_batch(batch)
            optimizer.zero_grad()
            output = model(**batch)
            if i == 0:
                print_batch(tokenizer, batch, output)
            batch_loss = output.loss.sum()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            del batch
        metrics = {"loss": epoch_loss}
        print(metrics)
        if run:
            run.log(metrics)
    print("finished training")


def print_batch(
    tokenizer: RagTokenizer, batch: BatchEncoding, output: ModelOutput
) -> None:
    """Decode and print the inputs and outputs of a batch, and the loss"""
    questions = tokenizer.question_encoder.batch_decode(
        batch["input_ids"], skip_special_tokens=True
    )
    answers = tokenizer.question_encoder.batch_decode(
        batch["labels"], skip_special_tokens=True
    )
    for q, a, loss in zip(questions, answers, output.loss):
        print(f"Q: {q}")
        print(f"A: [{loss}] {a} ")


def load_model(debug: bool = False) -> tuple[RagTokenizer, RagModel]:
    """Load the RAG model. If debug=True, use the dummy dataset."""
    print("loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    print("loading retriever...")
    if debug:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        )
    else:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq",
            index_name="custom",
            passages_path="/mnt/disks/data/wiki_dpr",
            index_path="/mnt/disks/data/wiki_dpr.faiss",
        )
    print("loading model...")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq", retriever=retriever
    )
    return tokenizer, model


def load_base_model(debug: bool = False) -> tuple[RagTokenizer, RagModel]:
    """Load the RAG model. If debug=True, use the dummy dataset.

    The question encoder can be any autoencoding model, preferably DPRQuestionEncoder,
    and the generator can be any seq2seq model, preferably BartForConditionalGeneration.

    The model can be initialized with a RagRetriever for end-to-end generation or
    used in combination with the outputs of a retriever in multiple steps---see
    examples for more details. The model is compatible any autoencoding model
    as the question_encoder and any seq2seq model with language model head as
    the generator. It has been tested with DPRQuestionEncoder as the question_encoder
    and BartForConditionalGeneration or T5ForConditionalGeneration as the generator.
    """
    print("loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

    print("loading retriever...")
    if debug:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        )
    else:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq",
            index_name="custom",
            passages_path="/mnt/disks/data/wiki_dpr",
            index_path="/mnt/disks/data/wiki_dpr.faiss",
        )

    # RagRetriever(config=None, )
    print("loading model...")
    model = RagTokenForGeneration(
        config=None,
        question_encoder=None,
        generator=None,
        retriever=retriever,
    )
    return tokenizer, model


def load_ms_marco(debug: bool = False) -> Dataset:
    """Load the MS-MARCO dataset. If debug=True, select a tiny subset."""
    dataset = load_dataset("ms_marco", "v1.1")
    # dataset = load_dataset("ms_marco", "v2.1")
    if debug:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))
        dataset["validation"] = dataset["validation"].select(range(100))
    print(dataset)
    return dataset


def get_preproc_function(tokenizer: RagTokenizer) -> Callable[[Dataset], BatchEncoding]:
    """Get the preprocessing function that will tokenize Dataset objects"""

    def preprocess(examples: Dataset) -> BatchEncoding:
        inputs = examples["query"]
        targets = [ex[0] if len(ex) > 0 else " " for ex in examples["answers"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True
        )
        return model_inputs

    return preprocess


def pad_batch(batch: BatchEncoding) -> None:
    """Given a seq2seq batch encoding, pad it in place."""
    batch["input_ids"] = pad_vectors(batch["input_ids"], 0)
    batch["labels"] = pad_vectors(batch["labels"], 0)
    batch["attention_mask"] = pad_vectors(batch["attention_mask"], 0)
    batch["token_type_ids"] = pad_vectors(batch["token_type_ids"], 0)


def pad_vectors(vectors: list[Tensor], pad: int) -> Tensor:
    """Given a list of vector tensors, pad them to the same length and stack them."""
    assert all(map(lambda x: len(x.shape) == 1, vectors))
    max_len = max(map(lambda x: x.shape[0], vectors))
    padded = map(lambda x: pad_vector(x, max_len, pad), vectors)
    return torch.stack(tuple(padded))


def pad_vector(vector: Tensor, length: int, pad: int) -> Tensor:
    """Given a vector tensor, extend it to the given length using `pad`."""
    # pylint: disable-next=not-callable
    return torch.nn.functional.pad(vector, (0, length - len(vector)), value=pad)


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
