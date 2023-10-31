import pytest
import pandas as pd


@pytest.fixture(scope='module')
def data():
    """Get customers data to feed into the tests"""
    return pd.read_csv("data.csv", index_col=[0])

@pytest.fixture(scope='module')
def data():
    """Get customers data_train to feed into the tests"""
    return pd.read_csv("application_train.csv", index_col=[0])

@pytest.fixture(scope='module')
def data():
    """Get customers data_test to feed into the tests"""
    return pd.read_csv("application_test.csv", index_col=[0])


def test_data_duplicates(data):
    """Test if the data duplicated dataframe is empty --> no duplicates"""
    duplicates = data[data.duplicated()]
    assert duplicates.empty


def test_data_target_col(data):
    """Test that the train dataframe has a 'target' column"""
    assert 'TARGET' in data.columns


def test_train_test_sizes(application_train, application_test):
    """verifions si le data d'entrainement et le data de production ont le même nombre de colonnes, sans le target"""
    train_size = application_train.drop(columns='TARGET').shape[1]
    test_size = application_test.shape[1]
    assert train_size == test_size