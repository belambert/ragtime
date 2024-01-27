import typer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    RagRetriever,
    RagTokenForGeneration,
    RagTokenizer,
)
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
CACHE_DIR = "/mnt/disks/data"  # ?


# this might help me load the index off of a read only volume
# >>> # To load your own indexed dataset built with the datasets
# >>> library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
# >>> from transformers import RagRetriever
# >>> dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
# >>> index_path = "path/to/my/index.faiss"  # faiss index saved via
# *dataset.get_index("embeddings").save(...)*
# >>> retriever = RagRetriever.from_pretrained(
# ...     "facebook/dpr-ctx_encoder-single-nq-base",
# ...     index_name="custom",
# ...     passages_path=dataset_path,
# ...     index_path=index_path,
# ... )


def main(debug: Annotated[bool, typer.Option()] = False):
    device = get_device()
    print(device)
    print("loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    print("loading retriever...")
    if debug:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        )
    else:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
        )
    print("loading model...")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq", retriever=retriever
    )

    dataset = load_dataset("ms_marco", "v1.1")
    if debug:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))
        dataset["validation"] = dataset["validation"].select(range(100))
    print(dataset)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def preprocess(examples):
        inputs = examples["query"]
        targets = [ex[0] if len(ex) > 0 else " " for ex in examples["answers"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True
        )
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    )
    tokenized_dataset.set_format("torch")

    # the dataloader isn't working though.
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        collate_fn=collator,
        batch_size=8,
    )
    # eval_dataloader = DataLoader(
    #     tokenized_dataset["test"], collate_fn=collator, batch_size=8
    # )

    for epoch in range(EPOCHS):
        print(f"epoch: {epoch}")
        # Training
        model.train()
        for i, batch in enumerate(train_dataloader):
            print(f"batch: {i}")
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


if __name__ == "__main__":
    typer.run(main)
