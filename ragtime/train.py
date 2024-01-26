import torch
import typer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
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


def main():
    device = get_device()
    print(device)
    print("loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    print("loading retriever...")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
    )
    print("loading model...")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq", retriever=retriever
    )

    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    # v2 is much larger
    dataset = load_dataset("ms_marco", "v1.1")
    # dataset = load_dataset("ms_marco", 'v2.1')
    dataset["train"] = dataset["train"].select(range(100))
    dataset["test"] = dataset["test"].select(range(100))
    dataset["validation"] = dataset["validation"].select(range(100))
    print(dataset)

    # def preprocess(examples):
    #     inputs = examples["query"]
    #     targets = [
    #         answers[0] if len(answers) > 0 else "" for answers in examples["answers"]
    #     ]
    #     model_inputs = tokenizer(
    #         inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True
    #     )
    #     return model_inputs

    # tokenized_data = dataset.map(
    #     preprocess,
    #     batched=True,
    #     remove_columns=dataset["train"].column_names,
    #     num_proc=8,
    # )
    # collator = DefaultDataCollator()
    # collator = DataCollatorForSeq2Seq(tokenizer) #, model=model)
    # print(tokenized_data)

    for epoch in range(EPOCHS):
        # for batch in tokenized_data["train"].iter(4, drop_last_batch=True):
        for batch in dataset["train"].iter(4, drop_last_batch=True):
            # print(batch)
            print("\n".join(batch["query"]))
            answers = list(map(lambda x: x[0], batch["answers"]))
            batch = tokenizer.prepare_seq2seq_batch(
                batch["query"], answers, return_tensors="pt"
            )

            print(tokenizer.question_encoder.batch_decode(batch["input_ids"], skip_special_tokens=True))
            # print(tokenizer.question_encoder.batch_decode(batch, skip_special_tokens=True))
            # print(batch)
            # result = model(*batch)
            # print(result)

            # 1. Encode
            print("encoding...")
            # question_hidden_states = model.question_encoder(batch["input_ids"])[0]
            question_hidden_states = model.question_encoder(batch)[0]
            # 2a. Retrieve the docs
            print("retrieving...")
            docs_dict = retriever(
                # batch["input_ids"].numpy(), question_hidden_states.detach().numpy(), return_tensors="pt"
                batch["input_ids"].numpy(), question_hidden_states.numpy(), return_tensors="pt"
            )
            # keys of docs_dict are: dict_keys(['context_input_ids',
            # 'context_attention_mask', 'retrieved_doc_embeds', 'doc_ids'])

            # 2b. compute the doc scores
            print("scoring...")
            doc_scores = torch.bmm(
                question_hidden_states.unsqueeze(1),
                docs_dict["retrieved_doc_embeds"].float().transpose(1, 2),
            ).squeeze(1)

            # if print_docs:
            #     _print_docs(docs_dict, doc_scores, dataset)

            # 3. generate
            generated = model.generate(
                context_input_ids=docs_dict[
                    "context_input_ids"
                ],  # the text of the retrieved docs
                context_attention_mask=docs_dict["context_attention_mask"],
                doc_scores=doc_scores,
            )
            print(tokenizer.batch_decode(generated, skip_special_tokens=True))




            # print(batch.keys())
            # print(batch["input_ids"])
            # print(len(batch["labels"]))
            # # print(torch.tensor(batch["input_ids"]))
            # print(collator(batch))
            break
    return

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # print("loading retriever...")
    # retriever = RagRetriever.from_pretrained(
    #     "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
    # )
    # print("loading model...")
    # model = RagTokenForGeneration.from_pretrained(
    #     "facebook/rag-token-nq", retriever=retriever
    # )
    # print(model)

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
        # compute_metrics=compute_metrics,
    )
    trainer.evaluate(max_length=MAX_LENGTH)
    trainer.train()
    # trainer.evaluate(max_length=MAX_LENGTH)


if __name__ == "__main__":
    typer.run(main)
