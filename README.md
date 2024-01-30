Ragtime
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

```console
> poetry run inference "what is a white oak?"
✔ generating...
Answer: quercus alba
```

Use the options `--citations` and `--sources` to see which wikipedia snippets were
used to generate the answer.

```console
> poetry run inference "what is a white oak?" --citations --sources
✔ generating...
Answer: quercus alba
Sources
"Oak (wine)" (68.10) - may spend two years. The very tannic Nebbiolo grape may spend four or more years in oak. High end Rioja producers will sometimes age their wines up to ten years in American oak to get a desired earthy cedar and herbal character. The species of oak typically used for American oak production is the "Quercus alba" which is a white oak species that is characterized by its relatively fast growth, wider grains and lower wood tannins. It is found in most of the Eastern United States as well as Missouri, Minnesota and Wisconsin where many wine barrels are from. In
Oak (67.53) - red oak. Cherrybark oak is another type of red oak which provides excellent timber. The standard for the lumber of the white oak group – all of which is marketed as white oak – is the "Quercus alba". White oak is often used to make wine barrels. The wood of the deciduous pedunculate oak and sessile oak accounts for most of the European oak production, but evergreen species, such as Holm oak and cork oak also produce valuable timber. The bark of the white oak is dried and used in medical preparations. Oak bark is also rich in tannin, and
Oak (67.48) - India, besides fuelwood and timber, the local people use oak wood for making agricultural implements. The leaves are used as fodder during lean period and bedding for livestock. The bark of the cork oak is used to produce wine stoppers (corks). This species grows in the Mediterranean Sea region, with Portugal, Spain, Algeria, and Morocco producing most of the world's supply. Of the North American oaks, the northern red oak is one of the most prized of the red oak group for lumber, much of which is marketed as red oak regardless of the species of origin. It is not
Oak (66.80) - oaks ("Quercus robur", "Q. petraea") give the wine greater refinement and are chosen for the best wines since they increase the price compared to those aged in American oak wood. American oak contributes greater texture and resistance to ageing, but produces more powerful wine bouquets. Oak wood chips are used for smoking fish, meat, cheeses, and other foods. Japanese oak is used in the making of professional drums from the manufacturer Yamaha Drums. The higher density of oak gives the drum a brighter and louder tone compared to traditional drum materials such as maple and birch. In hill states of
"Quercus montana" (66.06) - trees are generally not the best timber trees because they are usually branched low and not very straight, but when they grow in better conditions, they are valuable for timber, which is marketed as 'mixed white oak'. Quercus montana Quercus montana, the chestnut oak, is a species of oak in the white oak group, "Quercus" sect. "Quercus". It is native to the eastern United States, where it is one of the most important ridgetop trees from southern Maine southwest to central Mississippi, with an outlying northwestern population in southern Michigan. It is also sometimes called "rock oak" because of its
```


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



Data sets used in the RAG paper
===============================

<ins>Open domain QA</ins>

> Natural Questions (NQ) [29], TriviaQA (TQA) [24]. WebQuestions (WQ) [3] and CuratedTrec (CT) [2]. As
> CT and WQ are small, we follow DPR [26] by initializing CT and WQ models with our NQ RAG
> model. We use the same train/dev/test splits as prior work [31, 26] and report Exact Match (EM)
> scores. For TQA, to compare with T5 [52], we also evaluate on the TQA Wiki test set.

Huggingface dataset: `natural_questions`, `trivia_qa`, `web_questions`.

<ins>Abstractive Question Answering</ins>

> MSMARCO NLG task v2.1 [43]. The task consists of questions, ten gold passages
> retrieved from a search engine for each question, and a full sentence answer annotated from the
> retrieved passages. We do not use the supplied passages, only the questions and answers, to treat
> MSMARCO as an open-domain abstractive QA task. MSMARCO has some questions that cannot be
> answered in a way that matches the reference answer without access to the gold passages, such as
> “What is the weather in Volcano, CA?” so performance will be lower without using gold passages.
> We also note that some MSMARCO questions cannot be answered using Wikipedia alone. Here,
> RAG can rely on parametric knowledge to generate reasonable responses.

`ms_marco` on Huggingface.

Links
=====
- [Original RAG paper](https://arxiv.org/pdf/2005.11401.pdf)
- [Who cites RAG?](https://www.semanticscholar.org/paper/Retrieval-Augmented-Generation-for-NLP-Tasks-Lewis-Perez/58ed1fbaabe027345f7bb3a6312d41c5aac63e22#citing-papers)
- [More RAG papers](https://paperswithcode.com/method/rag)

TODO
====
- Option to fine-tune from base Rag model vs. the already fine-tuned model.
- Simplify the code for loading the various models from various locations.
- Capture some more Huggingface warnings.
- Save the trained model and upload to HF
- Parallel training
