
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
4
MSMARCO as an open-domain abstractive QA task. MSMARCO has some questions that cannot be
answered in a way that matches the reference answer without access to the gold passages, such as
“What is the weather in Volcano, CA?” so performance will be lower without using gold passages.
We also note that some MSMARCO questions cannot be answered using Wikipedia alone. Here,
RAG can rely on parametric knowledge to generate reasonable responses.


# takehome-ml

Use whatever tools you'd like to build a CLI for searching, analyzing, or otherwise interacting with some subset of English Wikipedia. The CLI can only take in a single text input sequence similar to a search box -- no programmatic filters or other options. Results can be in whatever form you like.

Example user queries:

- "Who is the president?"
- "Who scored the most goals in the European Champion's League in 2020?"
- "articles similar to the article on 'Philosophy'?"
- "the school of athens"
- "css flex"

Your goal should be to show off your ML skills and demonstrate a thorough analysis of the problem during the review session. If you're a data wizard, you can show off your deep familiarity with data processing, creating synthesized or manually annotated data as needed to bootstrap parts of your system. If you're a modeling expert, show us a deep understanding of the models you use and try to train or fine-tune a model for some part of the system. If you love experiments, get a few in and show us how you'd set up metrics and walk us through some comparisons. Play to your strengths.

To get you thinking, here are some resources you could consider using to bootstrap parts of the system you wouldn't naturally focus on:

- https://en.wikipedia.org/wiki/Wikipedia:Database_download
- https://wikipedia.readthedocs.io/en/latest/quickstart.html#quickstart
- [Download WikiText-2 raw character level data](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip)
- [Download WikiText-103 raw character level data](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip)
- https://huggingface.co/transformers/main_classes/pipelines.html
- https://github.com/facebookresearch/faiss
- https://colab.research.google.com/

When you're finished, upload any code that you used to setup or run your CLI tool to this repository with notes on how to get the program running locally. We'll try to run it before your review session, which you can schedule by sending an email to recruiting@you.com (cc bryan@you.com) with the subject "[ML Takehome Finished] YOUR_NAME" and your availability.
