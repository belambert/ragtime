
The original RAG implementation is a compelling natural language interface to
Wikipedia.  The [fine-tuned version available on
Huggingface](https://huggingface.co/facebook/rag-token-nq) appears to be fine
tuned on the [Natural Questions dataset](https://huggingface.co/datasets/natural_questions)
short answers, which means it typically gives very short responses, often
only 1-3 words long. However, wikipedia contains very rich and detailed information,
so a RAG model should be able to be fine-tuned to give more detailed answers. In
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

    poetry run python ./ragtime/inference.py "what is a sunflower?"

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
