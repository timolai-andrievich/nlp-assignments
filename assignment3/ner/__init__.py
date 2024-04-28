"""Predictors for named entity recognition task.

Usage example:
    predictor = NaivePredictor()
    predictor.fit(train_corpus, train_labels)
    pred = predictor(test_corpus)
"""
from .huggingface import HuggingfacePredictor
from .naive import NaivePredictor
from .ngrams import NGramPredictor
