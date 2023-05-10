from collections import Sequence
import logging
from typing import List

import pandas as pd

# It's OK to import static values from src
from src.config import DATA_DIR, TARGET

DATASET_LOCATION = DATA_DIR / 'australian_open.csv'


def get_test_datapoints(count: int = 1) -> List[dict]:
    """Return `count` observations taken randomly from the raw data.

    The returned value is a list of datapoint dictionnaries, ready to be
    transformed to a JSON object if necessary.
    """
    df = get_raw_dataframe()
    X = df.drop([TARGET], axis=1)

    return X.sample(count).to_dict(orient='records')


def get_raw_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATASET_LOCATION)
