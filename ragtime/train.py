import torch
import typer
from datasets import load_dataset
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
    model.train()
    model.context_encoder_training = True

    dataset = load_dataset("ms_marco", "v1.1")
    if debug:
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
        print(f"epoch: {epoch}")
        # for batch in tokenized_data["train"].iter(4, drop_last_batch=True):
        for batch in dataset["train"].iter(4, drop_last_batch=True):
            # print(batch)
            print("\n".join(batch["query"]))
            answers = list(map(lambda x: x[0], batch["answers"]))
            batch = tokenizer.prepare_seq2seq_batch(
                batch["query"], answers, return_tensors="pt"
            )

            print(
                tokenizer.question_encoder.batch_decode(
                    batch["input_ids"], skip_special_tokens=True
                )
            )
            del batch["token_type_ids"]
            print(batch.keys())
            print(model(*batch))

            # print("labels:")
            # print(batch["labels"])
            print(
                tokenizer.question_encoder.batch_decode(
                    batch["labels"], skip_special_tokens=True
                )
            )
            # print(tokenizer.question_encoder.batch_decode(batch, skip_special_tokens=True))
            # print(batch)
            # result = model(*batch)
            # print(result)

            # 1. Encode
            print("encoding...")
            question_hidden_states = model.question_encoder(batch["input_ids"])[0]
            # question_hidden_states = model.question_encoder(batch)[0]
            # 2a. Retrieve the docs
            print("retrieving...")
            docs_dict = retriever(
                batch["input_ids"].numpy(),
                question_hidden_states.detach().numpy(),
                return_tensors="pt"
                # batch["input_ids"].numpy(), question_hidden_states.numpy(), return_tensors="pt"
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
            # generated = model.generate(
            #     context_input_ids=docs_dict[
            #         "context_input_ids"
            #     ],  # the text of the retrieved docs
            #     context_attention_mask=docs_dict["context_attention_mask"],
            #     doc_scores=doc_scores,
            #     return_dict_in_generate=True,
            #     return_dict=True
            # )

            generated = model(
                context_input_ids=docs_dict["context_input_ids"],
                context_attention_mask=docs_dict["context_attention_mask"],
                doc_scores=doc_scores,
                return_dict_in_generate=True,
                return_dict=True,
                labels=batch["labels"],
                reduce_loss=False,
            )

            print(generated.loss)
            # print(tokenizer.batch_decode(generated, skip_special_tokens=True))

            # print(batch.keys())
            # print(batch["input_ids"])
            # print(len(batch["labels"]))
            # # print(torch.tensor(batch["input_ids"]))
            # print(collator(batch))
            break


if __name__ == "__main__":
    typer.run(main)
