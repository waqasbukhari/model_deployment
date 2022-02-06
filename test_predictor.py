import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from model_building.ml.data import process_data
from model_building.ml.model import compute_model_metrics, inference

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def X():
    data_path = "data/modified_census.csv"
    data = pd.read_csv(data_path)
    train, _ = train_test_split(data, test_size=0.20)
    X, _, _, _ = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X


@pytest.fixture(scope="module")
def y():
    data_path = "data/modified_census.csv"
    data = pd.read_csv(data_path)
    train, _ = train_test_split(data, test_size=0.20)
    _, y, _, _ = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return y


@pytest.fixture(scope="module")
def model(X, y):
    dummy = DummyClassifier()
    dummy.fit(X, y)
    return dummy


@pytest.fixture(scope="module")
def preds(model, X):
    preds = inference(model, X)
    return preds


def test_compute_model_metrics_count(preds, y):
    metrics = compute_model_metrics(y, preds)
    assert len(metrics) == 3


def test_compute_model_metrics_range(y, preds):
    metrics = compute_model_metrics(y, preds)
    result = map((lambda m: (m >= 0) & (m <= 1)), metrics)
    assert all(result)


def test_inference_shape(model, X):
    preds = inference(model, X)
    assert len(preds) == X.shape[0]


def test_inference_values(model, X):
    preds = inference(model, X)
    assert np.all((preds == 0) | (preds == 1))
