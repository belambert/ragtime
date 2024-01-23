import typer
from datasets import load_dataset
from torch.utils.data import DataLoader
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


def main(query: Annotated[str, typer.Argument()]):
    device = get_device()
    print(device)

    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

    # v2 is much larger
    dataset = load_dataset("ms_marco", "v1.1")
    # dataset = load_dataset("ms_marco", 'v2.1')
    print(dataset)
    # print(dataset["train"][0])
    print(dataset["train"][0]["query"])
    print(dataset["train"][0]["answers"])

    train = dataset["train"]

    train = train.map(lambda x: tokenizer(x["query"]), batched=True)
    print(train)

    train_loader = DataLoader(train, batch_size=16, shuffle=True)

    for batch in train_loader:
        print(batch)

    # don't change the tokenizer

    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
    )
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq", retriever=retriever
    )
    print(model)

    # things we need to tune...
    # model.question_encoder

    input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")
    print(input_dict)
    # print("generating...")
    # generated = model.generate(input_ids=input_dict["input_ids"], max_new_tokens=20)
    # print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


if __name__ == "__main__":
    typer.run(main)
