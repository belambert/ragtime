import torch
import typer
from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
from typing_extensions import Annotated

# from: https://huggingface.co/docs/transformers/model_doc/rag#transformers.RagTokenForGeneration


def main(
    query: Annotated[str, typer.Argument()],
    print_docs: Annotated[bool, typer.Option()] = True,
    debug: Annotated[bool, typer.Option()] = True,
):
    # load model and get the dataset so we can print the passages
    tokenizer, retriever, model = load_model()
    dataset = retriever.index.dataset

    # 0. tokenizer
    inputs = tokenizer(query, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if debug:
        print(f"{input_ids=}")

    # 1. Encode
    question_hidden_states = model.question_encoder(input_ids)[0]
    # 2a. Retrieve the docs
    docs_dict = retriever(
        input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt"
    )
    # keys of docs_dict are: dict_keys(['context_input_ids', 'context_attention_mask', 'retrieved_doc_embeds', 'doc_ids'])

    # 2b. compute the doc scores
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1),
        docs_dict["retrieved_doc_embeds"].float().transpose(1, 2),
    ).squeeze(1)

    if print_docs:
        _print_docs(docs_dict, doc_scores, dataset)

    # 3. generate
    generated = model.generate(
        context_input_ids=docs_dict[
            "context_input_ids"
        ],  # the text of the retrieved docs
        context_attention_mask=docs_dict["context_attention_mask"],
        doc_scores=doc_scores,
    )
    if debug:
        gen_ids = torch.flatten(generated).tolist()
        print(gen_ids)
        print(tokenizer.generator.convert_ids_to_tokens(gen_ids))

    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
    print(f"{generated_string=}")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
    )
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq", retriever=retriever
    )
    return tokenizer, retriever, model


def _print_docs(docs_dict, doc_scores, dataset):
    doc_ids = torch.flatten(docs_dict["doc_ids"]).tolist()
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
