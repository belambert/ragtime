import warnings

import evaluate
import numpy as np
import typer
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from typing_extensions import Annotated

# from: https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt

# If you are using a T5 model (more specifically, one of the t5-xxx checkpoints),
# the model will expect the text inputs to have a prefix indicating the task at
# hand, such as `translate: English to French:`.

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

MODEL = "Helsinki-NLP/opus-mt-en-fr"
MAX_LENGTH = 128
METRIC = evaluate.load("sacrebleu")


def main(debug: Annotated[bool, typer.Option()] = False):
    raw_dataset = load_dataset("kde4", lang1="en", lang2="fr", trust_remote_code=True)
    dataset = raw_dataset["train"].train_test_split(train_size=0.9, seed=20)
    if debug:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, return_tensors="pt")
    check_tokenizer(dataset, tokenizer)
    tokenized_dataset = dataset.map(
        get_preprocess_function(tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    )
    print(tokenized_dataset)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    inspect_batch(tokenized_dataset, collator)

    args = Seq2SeqTrainingArguments(
        "marian-finetuned-kde4-en-to-fr",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=get_compute_metrics_function(tokenizer),
    )

    print(trainer.evaluate(max_length=MAX_LENGTH))
    trainer.train()
    print(trainer.evaluate(max_length=MAX_LENGTH))


def inspect_batch(dataset, collator):
    print("inspecting_batch...")
    batch = collator([dataset["train"][i] for i in range(1, 3)])
    print(batch.keys())
    print(batch["input_ids"])
    print(batch["labels"])
    print(batch["decoder_input_ids"])


def get_preprocess_function(tokenizer):
    """Return a function that does the tokenization for you."""

    def preprocess(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["fr"] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True
        )
        return model_inputs

    return preprocess


def get_compute_metrics_function(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = METRIC.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    return compute_metrics


def check_tokenizer(dataset, tokenizer):
    print("checking tokenizer...")
    en_sentence = dataset["train"][1]["translation"]["en"]
    fr_sentence = dataset["train"][1]["translation"]["fr"]
    inputs = tokenizer(en_sentence, text_target=fr_sentence)
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
    print(tokenizer.convert_ids_to_tokens(inputs["labels"]))


if __name__ == "__main__":
    typer.run(main)
