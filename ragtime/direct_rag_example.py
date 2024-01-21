import torch
import typer
from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
from typing_extensions import Annotated

# from: https://huggingface.co/docs/transformers/model_doc/rag#transformers.RagTokenForGeneration


def main(query: Annotated[str, typer.Argument()]):
    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
    )
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-token-nq", retriever=retriever
    )

    inputs = tokenizer(query, return_tensors="pt")
    input_ids = inputs["input_ids"]
    # targets = tokenizer(
    #     text_target="In Paris, there are 10 million people.", return_tensors="pt"
    # )
    # labels = targets["input_ids"]
    # outputs = model(input_ids=input_ids, labels=labels)

    # # or use retriever separately
    # model = RagTokenForGeneration.from_pretrained(
    #     "facebook/rag-token-nq", use_dummy_dataset=True
    # )

    # 1. Encode
    question_hidden_states = model.question_encoder(input_ids)[0]
    # 2. Retrieve
    docs_dict = retriever(
        input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt"
    )

    dataset = retriever.index.dataset
    print(dataset)

    print(f"{docs_dict.keys()=}")
    print(f"{docs_dict=}")
    doc_ids = torch.flatten(docs_dict["doc_ids"])
    # batch matrix-matix product
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1),
        docs_dict["retrieved_doc_embeds"].float().transpose(1, 2),
    ).squeeze(1)
    print(f"{doc_ids=}")
    print(f"{doc_scores=}")
    for doc_id in doc_ids:
        doc_id = doc_id.item()
        print("*" * 60)
        print(doc_id)
        print(dataset[doc_id]["title"])
        print(dataset[doc_id]["text"])
    print("*" * 60)
    # 3. Forward to generator
    # outputs = model(
    #     context_input_ids=docs_dict["context_input_ids"],
    #     context_attention_mask=docs_dict["context_attention_mask"],
    #     doc_scores=doc_scores,
    #     decoder_input_ids=labels,
    # )
    # this is a big gnarly thing... RetrievAugLMMarginOutput
    # print(f"{outputs=}")

    # or directly generate
    generated = model.generate(
        context_input_ids=docs_dict["context_input_ids"],
        context_attention_mask=docs_dict["context_attention_mask"],
        doc_scores=doc_scores,
    )
    print(f"{generated=}")
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
    print(f"{generated_string=}")


if __name__ == "__main__":
    typer.run(main)
