import pytest
import pandas as pd


@pytest.fixture(scope='module')
def data():
    """Get customers data to feed into the tests"""
    return pd.read_csv("data.csv", index_col=[0])


def test_data_duplicates(data):
    """Test if the data duplicated dataframe is empty --> no duplicates"""
    duplicates = data[data.duplicated()]
    assert duplicates.empty


def test_data_target_col(data):
    """Test that the train dataframe has a 'target' column"""
    assert 'TARGET' in data.columns.tolist()


