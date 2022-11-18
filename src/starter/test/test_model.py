import pickle

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

from src.starter.ml.data import process_data
from src.starter.ml.model import compute_model_metrics, train_model, inference


@pytest.fixture
def my_data():
    data = pd.read_csv('./starter/data/census_clean.csv')
    return data

@pytest.fixture
def my_splitted_data(my_data):
    train, test = train_test_split(my_data, test_size=0.20)
    return train, test

@pytest.fixture
def my_processed_data(my_splitted_data):
    train, test = my_splitted_data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
    )
    return X_train, y_train, X_test, y_test

def test_processed_data_dims(my_processed_data):
    X_train, y_train, X_test, y_test = my_processed_data
    assert X_train.ndim == X_test.ndim
    assert y_test.ndim == 1 and y_test.ndim == 1

@pytest.fixture
def my_model(my_processed_data):
    X_train, y_train, X_test, y_test = my_processed_data
    model = train_model(X_train, y_train)
    return model

def test_if_model_fitted(my_model, my_processed_data):
    X_train, y_train, X_test, y_test = my_processed_data
    error_occured=False
    try:
        my_model.predict(X_test)
    except NotFittedError as e:
        error_occured=True
    assert error_occured==False


def test_on_good_metrics(my_model, my_processed_data):
    X_train, y_train, X_test, y_test = my_processed_data
    preds = inference(my_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f'precision={precision:.2f}, recall={recall:.2f}, fbeta={fbeta:.2f}')
    assert precision>0.7 and recall>0.6 and fbeta>0.6
