import pprint
import time

import numpy as np
import requests
import torch
import typer
import wikipedia
from halo import Halo
from huggingface_hub import login
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import Annotated

from wikibot.device import get_device
from wikibot.prompt import FEW_SHOT_PROMPT2, LLAMA2_PROMPT

# FULL_MODEL = "facebook/opt-2.7b"  # this one fits in 16GB GPU
# FULL_MODEL = "facebook/opt-6.7b"  # this works in 8bit mode
FULL_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # this seems to work in 8bit mode
SMALL_MODEL = "facebook/opt-125m"

URL = "https://en.wikipedia.org/w/api.php"

# TODO - use the smaller model to speed up generation





def cli(
    query: Annotated[str, typer.Argument()],
    small: Annotated[bool, typer.Option()] = False,
    n_docs: Annotated[int, typer.Option()] = 5,
):
    login(token="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")
    device = get_device()
    print(device)
    model, tokenizer = load_model(small, device)
    model.eval()
    hits = search_wiki(query, n_docs)

    for hit in hits:
        # print("*" * 80)
        if small:
            prompt = FEW_SHOT_PROMPT2.format(question=query, article=hit.summary)
        else:
            prompt = LLAMA2_PROMPT.format(question=query, article=hit.summary)
        title = colored(hit.title, "dark_grey", attrs=["underline"])
        spinner = Halo(text=f"Reading: {title}", spinner="dots")
        spinner.start()
        answer, score = generate(prompt, tokenizer, model, device)
        spinner.succeed()
        if is_good_answer(answer):
            print(f"From {title} ({score:.6f}):")
            print(f"{answer}\n")

def is_good_answer(answer: str):
    if "I don't know" in answer:
        return False
    if "I do not have an answer" in answer:
        return False
    if "I cannot answer" in answer:
        return False
    if "I do not know" in answer:
        return False
    if "not mentioned in the article" in answer:
        return False
    if answer.strip() == "":
        return False
    return True

def load_model(small, device):
    if small:
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL)
        model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL)
        model.to(device)
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        FULL_MODEL, torch_dtype=torch.float16, load_in_8bit=True
    )
    return model, tokenizer

def search_wiki(query: str, n_docs: int) -> list[wikipedia.WikipediaPage]:
    sess = requests.Session()
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srlimit": n_docs,
        "srsearch": query,
    }
    result = sess.get(url=URL, params=params)
    hits = result.json().get("query").get("search")
    pageids = [hit.get("pageid") for hit in hits]
    pages = []
    for pageid in pageids:
        try:
            pages.append(wikipedia.page(pageid=pageid))
        except wikipedia.exceptions.DisambiguationError:
            pass
    return pages


def generate(prompt: str, tokenizer, model, device) -> tuple[str, float]:
    tokenized = tokenizer(prompt, return_tensors="pt")
    tokenized.to(device)
    n_prompt_tokens = tokenized.input_ids.shape[1]
    # if n_prompt_tokens > 1000:
    #     print(f"WARNING - # prompt tokens: {n_prompt_tokens}")
    with torch.no_grad():
        generated = model.generate(
            tokenized.input_ids,
            max_new_tokens=100,
            attention_mask=tokenized.attention_mask,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=1,
        )
        tokens = generated.sequences

        logits = model(input_ids=tokens).logits
        logits = torch.squeeze(logits, dim=0)
        probs = torch.nn.functional.softmax(logits, dim=1)
        token_probs = get_token_probs(probs, torch.squeeze(tokens, dim=0))
        token_probs = token_probs[n_prompt_tokens:]
        avg_token_prob = torch.mean(token_probs).item()

    str_ = tokenizer.decode(tokens[0], skip_special_tokens=True)
    str_ = str_[len(prompt)-1 :]
    return str_, avg_token_prob


def get_token_probs(probs, index):
    list_ = []
    for prob, i in zip(probs, index):
        list_.append(prob[i])
    return torch.stack(list_)


if __name__ == "__main__":
    typer.run(cli)
