import typer
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    RagRetriever,
    RagTokenForGeneration,
    RagTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


# and:
# https://towardsdatascience.com/how-to-fine-tune-a-q-a-transformer-86f91ec92997
# https://stackoverflow.com/questions/75854700/how-to-fine-tune-a-huggingface-seq2seq-model-with-a-dataset-from-the-hub

MAX_LENGTH = 128
EPOCHS = 1


def main(debug: Annotated[bool, typer.Option()] = False):
    device = get_device()
    print(device)
    print("loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained(
        "facebook/rag-token-nq", cache_dir="/mnt/disks/data"
    )
    print("loading retriever...")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq",
        dataset="wiki_dpr",
        index_name="compressed",
        cache_dir="/mnt/disks/data",
    )
    print("loading model...")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq", retriever=retriever, cache_dir="/mnt/disks/data"
    )

    dataset = load_dataset("ms_marco", "v1.1")
    if debug:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))
        dataset["validation"] = dataset["validation"].select(range(100))
    print(dataset)

    def preprocess(examples):
        inputs = examples["query"]
        targets = [
            answers[0] if len(answers) > 0 else "" for answers in examples["answers"]
        ]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True
        )
        return model_inputs

    tokenized_data = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    )
    print(tokenized_data)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        "training_ragtime",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.evaluate(max_length=MAX_LENGTH)
    trainer.train()
    trainer.evaluate(max_length=MAX_LENGTH)


if __name__ == "__main__":
    typer.run(main)
