"""Module containing metrics for NER, as well as types for entity labels and
the base predictor class.

Usage example:
    predictor = NaivePredictor()
    predictor.fit(train_corpus, train_labels)
    pred = predictor(test_corpus)
    f1_score = char_wise_f1_score_macro(pred, target)
"""
from .metrics import char_wise_f1_score_macro
from .predictor import BasePredictor
from .types import Entity
