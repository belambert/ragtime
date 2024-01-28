import torch
import typer
from datasets import Dataset, load_dataset
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer
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
EPOCHS = 1


def main(debug: Annotated[bool, typer.Option()] = False):
    device = get_device()
    print(device)
    # pylint: disable-next=unused-variable
    tokenizer, retriever, model = load_model(debug)
    # tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    dataset = load_ms_marco(debug)

    # answers = [x[0] for x in dataset["train"][0:2]["answers"]]
    # batch0 = tokenizer.prepare_seq2seq_batch(
    #     dataset["train"][0:2]["query"], answers, return_tensors="pt"
    # )
    # print(batch0)

    tokenized_dataset = dataset.map(
        get_preproc_function(tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    )
    tokenized_dataset.set_format("torch")

    print("getting iterator...")
    data_iter = tokenized_dataset["train"].iter(4, drop_last_batch=True)

    print("beginning loop...")
    for i, batch in enumerate(data_iter):
        print(f"batch #{i}")
        pad_batch(batch)
        output = model(**batch)
        print(output.loss)

    # for epoch in range(EPOCHS):
    #     print(f"epoch: {epoch}")
    #     # for batch in tokenized_data["train"].iter(4, drop_last_batch=True):
    #     for batch in dataset["train"].iter(4, drop_last_batch=True):
    #         # print(batch)
    #         print("\n".join(batch["query"]))
    #         answers = list(map(lambda x: x[0], batch["answers"]))
    #         batch = tokenizer.prepare_seq2seq_batch(
    #             batch["query"], answers, return_tensors="pt"
    #         )
    #         print(batch)
    #         print(
    #             tokenizer.question_encoder.batch_decode(
    #                 batch["input_ids"], skip_special_tokens=True
    #             )
    #         )
    #         result = model(**batch)
    #         print(result.loss)
    #         # next thing is to do the weight update...
    #         break


def load_model(debug: bool = False):
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
    print(model)
    return tokenizer, retriever, model


def load_ms_marco(debug: bool = False) -> Dataset:
    dataset = load_dataset("ms_marco", "v1.1")
    # dataset = load_dataset("ms_marco", "v2.1")
    if debug:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))
        dataset["validation"] = dataset["validation"].select(range(100))
    print(dataset)
    return dataset


def get_preproc_function(tokenizer):
    def preprocess(examples):
        inputs = examples["query"]
        targets = [ex[0] if len(ex) > 0 else " " for ex in examples["answers"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True
        )
        return model_inputs

    return preprocess


def pad_batch(batch) -> None:
    batch["input_ids"] = pad_vectors(batch["input_ids"], 0)
    batch["labels"] = pad_vectors(batch["labels"], 0)
    batch["attention_mask"] = pad_vectors(batch["attention_mask"], 0)
    batch["token_type_ids"] = pad_vectors(batch["token_type_ids"], 0)


def pad_vectors(vectors: list[torch.Tensor], value) -> torch.Tensor:
    assert all(map(lambda x: len(x.shape) == 1, vectors))
    max_len = max(map(lambda x: x.shape[0], vectors))
    padded = map(lambda x: pad_vector(x, max_len, value), vectors)
    return torch.stack(tuple(padded))


def pad_vector(vector: torch.Tensor, length: int, value) -> torch.Tensor:
    # pylint: disable-next=not-callable
    return torch.nn.functional.pad(vector, (0, length - len(vector)), value=value)


if __name__ == "__main__":
    typer.run(main)
