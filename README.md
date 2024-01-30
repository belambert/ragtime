ragtime
=======
The [original RAG implementation](https://arxiv.org/pdf/2005.11401.pdf) is a
compelling natural language interface to Wikipedia.  The [fine-tuned version available on
Huggingface](https://huggingface.co/facebook/rag-token-nq) appears to be fine
tuned on the [Natural Questions dataset](https://huggingface.co/datasets/natural_questions)
short answers. This means it typically gives very short responses, often only
1-3 words long. However, wikipedia contains very rich and detailed information,
so we should be able to fine-tune a RAG model to give more detailed answers. In
this repo, I attempt to fine-tune the original RAG model to produce longer and
more detailed answers.

Setup
=====
If you don't already have python `poetry` installed, run this:

    pip install poetry

Once poetry is installed, you can set up a new virtual environment and install
wikibot by running:

    poetry install

Using
=====
To use, give queries on the command line, like this:

    poetry run inference "what is a marginal wood fern?"


Fine-tuning
===========
The `ragtime/train.py` script is set up to fine-tune the RAG model using the MS MARCO
dataset. The answers from this dataset appear to be a little longer than those from
Natural Questions. But this code could be modified to train from any dataset of
questions and answers.

To train, run:

    poetry run train

A few training parameters can be set with CLI option. Pass the `--help` flag to see
them.

```console
> poetry run train  --help
Usage: train [OPTIONS]

  Fine-tune a RAG model with the MS MARCO dataset.

Options:
  --debug / --no-debug  use data subset  [default: no-debug]
  --epochs INTEGER      num epochs  [default: 100]
  --lr FLOAT            learning rate  [default: 0.001]
  --batch-size INTEGER  batch size  [default: 4]
  --wandb / --no-wandb  use wandb  [default: no-wandb]
  --help                Show this message and exit.
```



-----

[Original RAG paper](https://arxiv.org/pdf/2005.11401.pdf)

[Who cites RAG?](https://www.semanticscholar.org/paper/Retrieval-Augmented-Generation-for-NLP-Tasks-Lewis-Perez/58ed1fbaabe027345f7bb3a6312d41c5aac63e22#citing-papers)

More RAG stuff: https://paperswithcode.com/method/rag


Test sets from the RAG paper:

Open domain QA:

Natural Questions (NQ) [29], TriviaQA (TQA) [24]. WebQuestions (WQ) [3] and CuratedTrec (CT) [2]. As
CT and WQ are small, we follow DPR [26] by initializing CT and WQ models with our NQ RAG
model. We use the same train/dev/test splits as prior work [31, 26] and report Exact Match (EM)
scores. For TQA, to compare with T5 [52], we also evaluate on the TQA Wiki test set.

natural_questions
trivia_qa
web_questions

ms_marco X


Abstractive Question Answering

MSMARCO NLG task v2.1 [43]. The task consists of questions, ten gold passages
retrieved from a search engine for each question, and a full sentence answer annotated from the
retrieved passages. We do not use the supplied passages, only the questions and answers, to treat
MSMARCO as an open-domain abstractive QA task. MSMARCO has some questions that cannot be
answered in a way that matches the reference answer without access to the gold passages, such as
“What is the weather in Volcano, CA?” so performance will be lower without using gold passages.
We also note that some MSMARCO questions cannot be answered using Wikipedia alone. Here,
RAG can rely on parametric knowledge to generate reasonable responses.
