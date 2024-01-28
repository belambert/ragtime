from typing import Callable

import torch
import typer
from datasets import Dataset, load_dataset
from torch import Tensor
from transformers import (
    BatchEncoding,
    RagModel,
    RagRetriever,
    RagTokenForGeneration,
    RagTokenizer,
)
from transformers.modeling_outputs import Seq2SeqModelOutput
from typing_extensions import Annotated

from ragtime.device import get_device

# Refer to:
# https://huggingface.co/datasets/wiki_dpr
# https://huggingface.co/facebook/rag-token-nq

# For Ray finetuning see:
# https://huggingface.co/blog/ray-rag
# https://shamanesiri.medium.com/how-to-finetune-the-entire-rag-architecture-including-dpr-retriever-4b4385322552
# https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/finetune_rag.py


MAX_LENGTH = 128
EPOCHS = 100
LR = 0.001


def main(debug: Annotated[bool, typer.Option()] = False) -> None:
    device = get_device()
    print(device)
    tokenizer, model = load_model(False)
    # tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    dataset = load_ms_marco(debug)

    tokenized_dataset = dataset.map(
        get_preproc_function(tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    )
    tokenized_dataset.set_format("torch")
    print(model.generator.parameters())
    print(model.question_encoder.parameters())
    # optimizer = torch.optim.AdamW(model.question_encoder.parameters(), lr=LR)
    optimizer = torch.optim.AdamW(model.generator.parameters(), lr=LR)

    print("beginning loop...")
    for _ in range(EPOCHS):
        data_iter = tokenized_dataset["train"].iter(4, drop_last_batch=True)
        for i, batch in enumerate(data_iter):
            print(f"batch #{i}")
            pad_batch(batch)
            optimizer.zero_grad()
            output = model(**batch)
            print_batch(tokenizer, batch, output)
            output.loss.sum(1).backward()
            optimizer.step()


def print_batch(
    tokenizer: RagTokenizer, batch: BatchEncoding, output: Seq2SeqModelOutput
) -> None:
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


def load_ms_marco(debug: bool = False) -> Dataset:
    dataset = load_dataset("ms_marco", "v1.1")
    # dataset = load_dataset("ms_marco", "v2.1")
    if debug:
        dataset["train"] = dataset["train"].select(range(4))
        dataset["test"] = dataset["test"].select(range(4))
        dataset["validation"] = dataset["validation"].select(range(4))
    print(dataset)
    return dataset


def get_preproc_function(tokenizer: RagTokenizer) -> Callable[[Dataset], BatchEncoding]:
    def preprocess(examples: Dataset) -> BatchEncoding:
        inputs = examples["query"]
        targets = [ex[0] if len(ex) > 0 else " " for ex in examples["answers"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True
        )
        return model_inputs

    return preprocess


# def pad_batch(batch: dict[str, Tensor]) -> None:
def pad_batch(batch: BatchEncoding) -> None:
    batch["input_ids"] = pad_vectors(batch["input_ids"], 0)
    batch["labels"] = pad_vectors(batch["labels"], 0)
    batch["attention_mask"] = pad_vectors(batch["attention_mask"], 0)
    batch["token_type_ids"] = pad_vectors(batch["token_type_ids"], 0)


def pad_vectors(vectors: list[Tensor], pad: int) -> Tensor:
    """
    Given a list of vector tensors, pad them to the same length and stack them.
    """
    assert all(map(lambda x: len(x.shape) == 1, vectors))
    max_len = max(map(lambda x: x.shape[0], vectors))
    padded = map(lambda x: pad_vector(x, max_len, pad), vectors)
    return torch.stack(tuple(padded))


def pad_vector(vector: Tensor, length: int, pad: int) -> Tensor:
    """
    Given a vector tensor, extend it to the given length using `pad`.
    """
    # pylint: disable-next=not-callable
    return torch.nn.functional.pad(vector, (0, length - len(vector)), value=pad)


if __name__ == "__main__":
    typer.run(main)
