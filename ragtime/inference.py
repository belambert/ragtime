import torch
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
    print(dataset)
    print("loading model...")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq",
        retriever=retriever,
    )
    print("preparing input...")
    input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")
    print(input_dict)

    print("generating...")
    # returns RetrievAugLMOutput ?
    result = model(**input_dict, output_retrieved=True)
    print(result.logits)
    _print_docs(result.retrieved_doc_ids, result.doc_scores, dataset)
    # get argmax over logits, then decode
    # print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


def _print_docs(doc_ids, doc_scores, dataset):
    doc_ids = torch.flatten(doc_ids).tolist()
    docs = list(zip(doc_ids, torch.flatten(doc_scores).tolist()))
    docs.sort(key=lambda x: x[1], reverse=True)
    for id_, score in docs:
        print("*" * 60)
        print(f"__{id_}__({score})")
        print(dataset[id_]["title"])
        print(dataset[id_]["text"])
    print("*" * 60)


if __name__ == "__main__":
    typer.run(main)
