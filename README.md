

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
