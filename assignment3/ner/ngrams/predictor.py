from collections import defaultdict, Counter
import re

import tqdm.auto as tqdm

from ner.utils import BasePredictor, Entity

class NGramPredictor(BasePredictor):
    """Predictor for the named entity recognition task. Predicts the most common
    label for a N-gram.
    """

    def __init__(
        self,
        n: int,
        default_label: str = 'O',
        pattern: str = r'\S+',
    ):
        """Initializes the predictor and fits it on the provided data.

        Args:
            n (int): Number of tokens to use in N-Grams.
            default_label (str): Label that is given to the words by default.
                Defaults to "O".
            pattern (str): Regular expression pattern used to tokenize texts.
                Defaults to \\S+
        """

        self.n = n
        self.default_label = default_label
        self.pattern = re.compile(pattern)
        self.prefix_label_counts = defaultdict(Counter)
        self.suffix_label_counts = defaultdict(Counter)

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

        def does_overlap(
            span_a: tuple[int, int],
            span_b: tuple[int, int],
        ) -> bool:
            a, b = span_a
            x, y = span_b
            return a <= x <= b or a <= y <= b

        if len(train_texts) != len(train_labels):
            raise ValueError('The lengths of texts and labels differ.')

        # Collect label counts from the training data
        for text, labels in tqdm.tqdm(
                zip(train_texts, train_labels),
                total=len(train_texts),
                disable=not verbose,
                desc='Fitting on the training data',
        ):
            words = list(re.finditer(self.pattern, text))
            for i, word in enumerate(words):
                prefix = tuple(word.group()
                               for word in words[max(0, i - self.n + 1):i + 1])
                suffix = tuple(word.group() for word in words[i:i + self.n])
                overlap_flag = False
                for *label_span, label in labels:
                    if not does_overlap(word.span(), label_span):
                        continue
                    overlap_flag = True
                    self.prefix_label_counts[prefix].update([label])
                    self.suffix_label_counts[suffix].update([label])
                if not overlap_flag:
                    self.prefix_label_counts[prefix].update(
                        [self.default_label])
                    self.suffix_label_counts[suffix].update(
                        [self.default_label])

    def get_label(
        self,
        prefix_gram: tuple[str],
        suffix_gram: tuple[str],
    ) -> str:
        """Predicts a label for a group of tokens.

        Args:
            prefix_gram (tuple[str]): Target word and the preceding context.
            suffix_gram (tuple[str]): Target word and the succeding context.
        
        Returns:
            str: Most likely label for the token.
        """
        prefix_gram = tuple(prefix_gram)
        suffix_gram = tuple(suffix_gram)
        joint_counter = Counter()
        joint_counter.update(self.prefix_label_counts[prefix_gram])
        joint_counter.update(self.suffix_label_counts[suffix_gram])
        if len(joint_counter) < 1:
            return self.default_label
        (most_common_label, _count), = joint_counter.most_common(1)
        return most_common_label

    def predict(self, text: str) -> list[Entity]:
        """Finds entities in the text.

        Args:
            text (str): Text to search for entities in.
        
        Returns:
            list[Entity]: List of found entities.
        """
        result = []
        words = list(re.finditer(self.pattern, text))
        for i, word in enumerate(words):
            prefix = tuple(word.group()
                      for word in words[max(0, i - self.n + 1):i + 1])
            suffix = tuple(word.group() for word in words[i:i + self.n])
            label = self.get_label(prefix, suffix)
            if label == self.default_label:
                continue
            result.append((word.start(), word.end() - 1, label))
        return result