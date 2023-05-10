import pytest
from src.train.data import load_raw_data
import pandas as pd
from tests.integration.constants import DATA_COLUMNS, COLUMN_TYPES


@pytest.fixture()
def tennis_data():
    return load_raw_data()


def test_read_input_data(tennis_data):
    assert isinstance(tennis_data, pd.DataFrame)


def test_data_shape(tennis_data):
    assert tennis_data.shape[1] == 26


def test_column_names(tennis_data):
    assert all(column in DATA_COLUMNS for column in tennis_data.columns)


def test_data_schema(tennis_data):
    assert all(tennis_data.loc[:, col_name].dtype == COLUMN_TYPES[col_name] for col_name in tennis_data.columns)
