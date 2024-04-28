# Assignment 3

[Report](report/report.pdf)

## How to run

To generate submission files, run `python3 generate_test_predictions.py`. To get metrics, run/see [`metrics.ipynb`](metrics.ipynb) as a Jupyter notebook.

Fine-tuned [BERT](tandrievich/google-bert-finetuned) and [LaBSE](https://huggingface.co/tandrievich/LaBSE-finetuned) models are also available through HuggingFace API, and can be tried online on corresponding pages.

## Repository structure

- `data/` - data used for training/testing.
- `ner/` - code for the solutions.
    - `ner/naive/` - baseline solution.
    - `ner/ngrams/` - N-gram based solution. Both bigrams and trigrams submissions are generated using this code.
    - `ner/hugginface/` - Transformers-based solution. Both LaBSE and BERT submissions are generated using this code.
    - `ner/utils/` - Utility functions and classes.
- `report/` - PDF and LaTEX report files, as well as a simple script to compile TEX into PDF.

## Fine-tuning BERT and LaBSE

As fine-tuning is rather computationally expensive, the models are fine-tuned using Kaggle. The fine-tuning code is in [ner/huggingface/train.ipynb](ner/huggingface/train.ipynb).
