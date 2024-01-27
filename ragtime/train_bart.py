import typer
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from typing_extensions import Annotated

MAX_LENGTH = 128


def main(debug: Annotated[bool, typer.Option()] = False):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
    # summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    # gen = tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
    #                              clean_up_tokenization_spaces=False)[0]
    # print(gen)

    dataset = load_dataset("ms_marco", "v1.1")
    if debug:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))

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

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        "training_bart",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
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
        # compute_metrics=compute_metrics,
    )
    eval_ = trainer.evaluate(max_length=MAX_LENGTH)
    print(eval_)
    trainer.train()
    eval_ = trainer.evaluate(max_length=MAX_LENGTH)
    print(eval_)


if __name__ == "__main__":
    typer.run(main)
