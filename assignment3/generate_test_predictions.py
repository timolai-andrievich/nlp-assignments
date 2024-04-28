import pathlib
import zipfile

import pandas as pd
import torch

from ner.utils import BasePredictor
from ner import (
    HuggingfacePredictor,
    NaivePredictor,
    NGramPredictor,
)


def get_predictions(dataframe: pd.DataFrame,
                    predictor: BasePredictor) -> pd.DataFrame:
    """Predicts entities for the dataset, and returns the dataframe with
    predictions in the `ners` column.

    Args:
        dataframe (DataFrame): Pandas dataframe with texts and ids. Should
            contain two columns:
                id (int): ID of the corresponding text.
                sentences (str): Text to find entities in.

    Returns:
        DataFrame: Pandas dataframe with three columns:
            id (int): ID of the text
            ners (list[tuple[int, int, str]]): List of found entities, given
                as (start, end, label).
            sentences (str): Text.
    """
    result = []
    # Perform NER for each text in the dataset
    for id_, text in zip(dataframe['id'], dataframe['sentences']):
        ners = predictor.predict(text)
        result.append((id_, text, ners))
    # Store the results in a dataframe
    res_df = pd.DataFrame(result, columns=['id', 'sentences', 'ners'])
    return res_df


def main():
    """Entry point of the program.
    """
    # Determine whether the CUDA backend is available or not
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize predictors
    predictors: dict[str, BasePredictor] = {
        'huggingface': HuggingfacePredictor(device=device),
        'naive': NaivePredictor(),
        '2gram': NGramPredictor(2),
    }
    # Read test and train data
    train_df = pd.read_json('data/train.jsonl', lines=True)
    test_df = pd.read_json(
        'data/test.jsonl',
        lines=True,
    ).rename(columns={'senences': 'sentences'})  # pylint: disable=no-member
    # Fit predictors on the train data
    for predictor in predictors.values():
        predictor.fit(train_df['sentences'], train_df['ners'], verbose=True)
    # Predict labels for each predictors, and store them in a zip file for
    # the CodaLab submission
    for name, predictor in predictors.items():
        # Create a new directory for storing predictions
        file_dir = pathlib.Path(f'test_predictions/{name}_test')
        file_dir.mkdir(exist_ok=True, parents=True)
        # Perform NER
        pred_df = get_predictions(test_df, predictor)
        # Store predictions in an archive
        with zipfile.ZipFile(file_dir / 'test.zip', 'w') as archive:
            with archive.open('test.jsonl', 'w') as file:
                pred_df.to_json(file, lines=True, orient='records')


if __name__ == '__main__':
    main()
