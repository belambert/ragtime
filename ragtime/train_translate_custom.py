# custom training loop:
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

# from: https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt

MODEL = "Helsinki-NLP/opus-mt-en-fr"
MAX_LENGTH = 128
METRIC = evaluate.load("sacrebleu")


# pylint: disable-next=too-many-locals, too-many-statements
def main():
    raw_dataset = load_dataset("kde4", lang1="en", lang2="fr", trust_remote_code=True)
    dataset = raw_dataset["train"].train_test_split(train_size=0.9, seed=20)
    dataset["train"] = dataset["train"].select(range(100))
    dataset["test"] = dataset["test"].select(range(100))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, return_tensors="pt")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def preprocess(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["fr"] for ex in examples["translation"]]
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

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        collate_fn=collator,
        batch_size=8,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["test"], collate_fn=collator, batch_size=8
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    output_dir = "marian-finetuned-kde4-en-to-fr-accelerate"
    # model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
    # repo_name = get_full_repo_name(model_name)
    # repo_name
    # repo = Repository(output_dir, clone_from=repo_name)

    def postprocess(predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=128,
                )
            labels = batch["labels"]
            # Necessary to pad predictions and labels for being gathered
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            predictions_gathered = accelerator.gather(generated_tokens)
            labels_gathered = accelerator.gather(labels)
            decoded_preds, decoded_labels = postprocess(
                predictions_gathered, labels_gathered
            )
            METRIC.add_batch(predictions=decoded_preds, references=decoded_labels)

        results = METRIC.compute()
        print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            # repo.push_to_hub(
            #     commit_message=f"Training in progress epoch {epoch}", blocking=False
            # )


# using the model
# from transformers import pipeline

# # Replace this with your own checkpoint
# model_checkpoint = "huggingface-course/marian-finetuned-kde4-en-to-fr"
# translator = pipeline("translation", model=model_checkpoint)
# translator("Default to expanded threads")

if __name__ == "__main__":
    main()
