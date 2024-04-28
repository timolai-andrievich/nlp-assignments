"""Module for named entity recognition via naive approach (predicting the most
common label for each word.)
"""
from collections import defaultdict, Counter
import re

import tqdm.auto as tqdm

from ner.utils import BasePredictor, Entity


class NaivePredictor(BasePredictor):
    """Predictor for the named entity recognition task. Predicts the most common
    label for every token.
    """

    def __init__(
        self,
        default_label: str = 'O',
        pattern: str = r'\S+',
    ):
        """Initializes the predictor and fits it on the provided data.

        Args:
            default_label (str): Label that is given to the words by default.
                Defaults to "O".
            pattern (str): Regular expression pattern used to tokenize texts.
                Defaults to \\S+
        """

        self.default_label = default_label
        self.pattern = re.compile(pattern)
        self.label_counts = defaultdict(Counter)

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[list[Entity]],
        verbose: bool = True,
    ):
        """Fits the predictor on the training data.

        Args:
            train_texts (list[str]): The dataset of the training texts.
            train_labels (list[str]): Corresponding labels.
            verbose (bool): Whether to display the progress bar or not.
        """
        if len(train_texts) != len(train_labels):
            raise ValueError('The lengths of texts and labels differ.')

        def does_overlap(
            span_a: tuple[int, int],
            span_b: tuple[int, int],
        ) -> bool:
            """Determines whether two ranges overlap or not.
    
            Args:
                span_a (tuple[int, int]): First range.
                span_b (tuple[int, int]): Second range.
            
            Returns:
                bool: Whether the spans overlap or not.
            """
            a, b = span_a
            x, y = span_b
            return a <= x <= b or a <= y <= b

        # Collect label counts from the training data
        for text, labels in tqdm.tqdm(
                zip(train_texts, train_labels),
                total=len(train_texts),
                disable=not verbose,
                desc='Fitting on the training data',
        ):
            for word in re.finditer(self.pattern, text):
                overlap_flag = False
                # Count all labels with which the word overlaps
                for *label_span, label in labels:
                    if not does_overlap(word.span(), label_span):
                        continue
                    overlap_flag = True
                    self.label_counts[word.group()].update([label])
                # If the word doesn't overlap with any labels, label it with
                # the default label
                if not overlap_flag:
                    self.label_counts[word.group()].update(
                        [self.default_label])

    def get_label(self, token: str) -> str:
        """Predicts a label for a singular token.

        Args:
            token (str): Token to predict the label for.
        
        Returns:
            str: Most likely label for the token.
        """
        if token not in self.label_counts:
            return self.default_label
        (most_common_label, _count), = self.label_counts[token].most_common(1)
        return most_common_label

    def predict(self, text: str) -> list[Entity]:
        """Finds entities in the text.

        Args:
            text (str): Text to search for entities in.
        
        Returns:
            list[Entity]: List of found entities.
        """
        result = []
        # Predict label for each word in the text
        for token in re.finditer(self.pattern, text):
            label = self.get_label(token.group())
            # Do not include default labels in the final result
            if label == self.default_label:
                continue
            result.append((token.start(), token.end() - 1, label))
        return result
