from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from ner.utils import BasePredictor, Entity


class HuggingfacePredictor(BasePredictor):
    """Predictor for the named entity recognition task. Uses Huggingface
    model for token classification for predictions.
    """

    def __init__(
        self,
        model_id='tandrievich/LaBSE-finetuned',
        device='cpu',
    ):
        """Initializes the predictor and fits it on the provided data.

        Args:
            model_id (str): ID of the model to pass to from_pretrained methods.
            device (str): Device to run the model on. Defaults to 'cpu'
        """
        self.model_id = model_id
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.pipeline = pipeline(
            'ner',
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            aggregation_strategy='first',
        )

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
        # Model should be fine-tuned already, as fitting a big model is
        # computationally intensive

    def predict(self, text: str) -> list[Entity]:
        """Finds entities in the text.

        Args:
            text (str): Text to search for entities in.
        
        Returns:
            list[Entity]: List of found entities.
        """
        result = []
        pipeline_results = self.pipeline(text)
        for res in pipeline_results:
            result.append((res['start'], res['end'], res['entity_group']))
        return result
