"""Module containing the base class for named entity recognition predictors.
"""
from .types import Entity


class BasePredictor:
    """Predictor for the named entity recognition task.
    """

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
        raise NotImplementedError('fit functionality should be'
                                  ' implemented in subclasses')

    def predict(self, text: str) -> list[Entity]:
        """Finds entities in the text.

        Args:
            text (str): Text to search for entities in.
        
        Returns:
            list[Entity]: List of found entities.
        """
        raise NotImplementedError('predict functionality should be'
                                  ' implemented in subclasses')

    def _aggregate_predictions(self, preds: list[Entity]) -> list[Entity]:
        """Aggregates multiple subsequent predictions of one entity type into
        one prediction.

        Args:
            preds (list[Entity]): Predictions to aggregate.
        
        Returns:
            list[Entity]: Aggregated predictions.
        """
        # Sort the predictions by the starting character index
        preds.sort(key=lambda x: x[0])
        res = []
        for start, end, label in preds:
            # Add the first prediction without further processing
            if not res:
                res.append((start, end, label))
                continue
            # Merge two predictions if they are one characters apart (assumed
            # to be space,) and have the same label.
            p_start, p_end, p_label = res[-1]
            if p_label == label and p_end == start - 2:
                res.pop(-1)
                res.append((p_start, end, label))
        return preds

    def __call__(self, text: str) -> list[Entity]:
        """Predicts the entities using `predict`, aggregates the predictions,
        and returns the aggregated predictions.

        Args:
            text (str): Text to search for entities in.
        
        Returns:
            list[Entity]: List of found entities.
        """
        predictions = self.predict(text)
        predictions = self._aggregate_predictions(predictions)
        return predictions
