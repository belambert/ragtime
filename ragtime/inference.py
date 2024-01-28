import typer
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer
from typing_extensions import Annotated

from ragtime.device import get_device


def main(query: Annotated[str, typer.Argument()]):
    """
    Do RAG inference on the given query.
    """
    device = get_device()
    print(device)

    print("loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained(
        "facebook/rag-token-nq",
    )
    print("loading retriever...")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq",
        index_name="custom",
        passages_path="/mnt/disks/data/wiki_dpr",
        index_path="/mnt/disks/data/wiki_dpr.faiss",
    )
    dataset = retriever.index.dataset
    print("loading model...")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq",
        retriever=retriever,
    )
    print("preparing input...")
    input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")
    print(input_dict)

    print("generating...")
    # return RetrievAugLMOutput ?
    # may have to do forward pass instead?
    generated = model.generate(input_ids=input_dict["input_ids"], max_new_tokens=20)
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
    print(type(generated))
    print(generated)
    print(dataset)
    # print(generated.doc_scores)
    # print(generated.retrieved_doc_ids)
    result = model(**input_dict)
    print(result)
    print(result.doc_scores)
    print(result.retrieved_doc_ids)
    print(result.logits)


if __name__ == "__main__":
    typer.run(main)
