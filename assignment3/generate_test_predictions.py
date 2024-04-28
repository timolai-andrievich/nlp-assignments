import pathlib
import zipfile

import pandas as pd

from ner.utils import BasePredictor
from ner import (
    # HuggingfacePredictor,
    NaivePredictor,
    NGramPredictor,
)


def get_predictions(dataframe: pd.DataFrame,
                    predictor: BasePredictor) -> pd.DataFrame:
    result = []
    for id_, text in zip(dataframe['id'], dataframe['sentences']):
        ners = predictor.predict(text)
        result.append((id_, text, ners))
    res_df = pd.DataFrame(result, columns=['id', 'sentences', 'ners'])
    return res_df


def main():
    predictors: dict[str, BasePredictor] = {
        # 'huggingface': HuggingfacePredictor(),
        'naive': NaivePredictor(),
        '2gram': NGramPredictor(2),
    }
    train_df = pd.read_json('data/train.jsonl', lines=True)
    test_df = pd.read_json(
        'data/test.jsonl',
        lines=True,
    ).rename(columns={'senences': 'sentences'})
    for predictor in predictors.values():
        predictor.fit(train_df.sentences, train_df.ners, verbose=True)
    for name, predictor in predictors.items():
        pred_df = get_predictions(test_df, predictor)
        file_dir = pathlib.Path(f'test_predictions/{name}_test')
        file_dir.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(file_dir / 'test.zip', 'w') as archive:
            with archive.open('test.jsonl', 'w') as file:
                pred_df.to_json(file, lines=True, orient='records')


if __name__ == '__main__':
    main()
