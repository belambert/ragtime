import typer
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer
from typing_extensions import Annotated

from wikibot.device import get_device

# Refer to:
# https://huggingface.co/datasets/wiki_dpr
# https://huggingface.co/facebook/rag-token-nq


def main(query: Annotated[str, typer.Argument()]):
    device = get_device()
    print(device)

    print("loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    print("loading retriever...")
    # retriever = RagRetriever.from_pretrained(
    #     "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
    # )
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
    )
    print("loading model...")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq", retriever=retriever
    )
    print("preparing input...")
    input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")

    print("generating...")
    generated = model.generate(input_ids=input_dict["input_ids"], max_new_tokens=5)
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


if __name__ == "__main__":
    typer.run(main)
