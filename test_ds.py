import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest
from sklearn.metrics import roc_auc_score

import ds


# -------------------------
# Helper: normalize target to 0/1
# -------------------------

def to_binary_01(y):
    """
    Convert y to 0/1 if it is:
      - already numeric with two unique values
      - strings like 'n'/'y'
    """
    y_series = pd.Series(y).copy()

    # If object/string: try mapping common values
    if y_series.dtype == "object" or pd.api.types.is_string_dtype(y_series):
        vals = set(y_series.dropna().unique())
        if vals.issubset({"n", "y"}):
            return y_series.map({"n": 0, "y": 1}).astype(int)
        if vals.issubset({"N", "Y"}):
            return y_series.map({"N": 0, "Y": 1}).astype(int)

    # If numeric: force to {0,1} by sorting unique values
    if pd.api.types.is_numeric_dtype(y_series):
        uniq = sorted(set(y_series.dropna().unique()))
        if len(uniq) == 2:
            mapping = {uniq[0]: 0, uniq[1]: 1}
            return y_series.map(mapping).astype(int)

    raise AssertionError(
        f"Could not coerce target to binary 0/1. dtype={y_series.dtype}, uniques={y_series.unique()[:10]}"
    )


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture(scope="session")
def spam_df():
    df = ds.load_spam_data("data/spam.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "yesno" in df.columns, "Expected column 'yesno' in spam.csv"
    return df


@pytest.fixture(scope="session")
def encoded_df(spam_df):
    df_enc = ds.encode_categoricals(spam_df)
    assert isinstance(df_enc, pd.DataFrame)
    assert df_enc.shape == spam_df.shape
    assert "yesno" in df_enc.columns
    return df_enc


@pytest.fixture(scope="session")
def split_data(encoded_df):
    X, y = ds.split_features_target(encoded_df, target="yesno")
    y01 = to_binary_01(y)

    X_train, X_test, y_train, y_test = ds.train_test_split_data(
        X, y01, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def pca_data(split_data):
    X_train, X_test, y_train, y_test = split_data
    X_train_pca, X_test_pca, pca = ds.apply_pca(X_train, X_test, n_components=2)
    return X_train_pca, X_test_pca, y_train, y_test, pca


@pytest.fixture(scope="session")
def trained_model(pca_data):
    X_train_pca, _, y_train, _, _ = pca_data
    model = ds.train_decision_tree(X_train_pca, y_train, random_state=42)
    return model


# -------------------------
# Tests: preprocessing robustness
# -------------------------

def test_encode_categoricals_target_is_binary(encoded_df):
    y01 = to_binary_01(encoded_df["yesno"])
    vals = set(y01.unique())
    assert vals.issubset({0, 1}), f"Expected binary target after normalization, got {vals}"


def test_split_features_target_shapes(encoded_df):
    X, y = ds.split_features_target(encoded_df, target="yesno")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, (pd.Series, np.ndarray, list))
    assert "yesno" not in X.columns
    assert len(X) == len(y)


# -------------------------
# Tests: PCA
# -------------------------

def test_apply_pca_two_components(pca_data):
    X_train_pca, X_test_pca, y_train, y_test, pca = pca_data
    assert isinstance(X_train_pca, np.ndarray)
    assert isinstance(X_test_pca, np.ndarray)
    assert X_train_pca.shape[1] == 2
    assert X_test_pca.shape[1] == 2

    assert hasattr(pca, "explained_variance_ratio_")
    assert len(pca.explained_variance_ratio_) == 2


# -------------------------
# Tests: model training
# -------------------------

def test_train_decision_tree_predicts(trained_model, pca_data):
    _, X_test_pca, _, y_test, _ = pca_data
    preds = trained_model.predict(X_test_pca)
    assert len(preds) == len(y_test)


# -------------------------
# Tests: decisionplot is truly a decision boundary plot
# -------------------------

def test_decisionplot_calls_predict_on_grid(trained_model, pca_data, monkeypatch):
    """
    Autograder-proof check:
    - decisionplot must call model.predict() on a large grid
    - not just on the original points
    """
    X_train_pca, _, y_train, _, _ = pca_data
    X_plot = pd.DataFrame(X_train_pca, columns=["PC1", "PC2"])

    calls = {"n_rows": []}

    original_predict = trained_model.predict

    def wrapped_predict(X):
        X_arr = np.asarray(X)
        calls["n_rows"].append(X_arr.shape[0])
        return original_predict(X)

    monkeypatch.setattr(trained_model, "predict", wrapped_predict)

    ds.decisionplot(trained_model, X_plot, y_train, resolution=0.2)
    plt.close("all")

    assert len(calls["n_rows"]) >= 1, "decisionplot() never called model.predict()"

    # The key: at least one predict call must be MUCH larger than training size
    max_rows = max(calls["n_rows"])
    n_train = len(X_plot)

    assert max_rows > n_train * 3, (
        "decisionplot() should predict over a mesh grid (many points). "
        f"Got max predict rows={max_rows}, training rows={n_train}. "
        "This suggests you may not be generating a grid."
    )


# -------------------------
# Tests: evaluation metrics
# -------------------------

def test_evaluate_classification_keys_and_ranges(trained_model, pca_data):
    _, X_test_pca, _, y_test, _ = pca_data
    metrics = ds.evaluate_classification(trained_model, X_test_pca, y_test)

    assert isinstance(metrics, dict)
    required = ["accuracy", "precision", "recall", "f1"]
    for k in required:
        assert k in metrics, f"Missing metric '{k}'"
        assert 0.0 <= float(metrics[k]) <= 1.0, f"Metric {k} out of range: {metrics[k]}"


# -------------------------
# Tests: ROC/AUC must use predict_proba (not predict)
# -------------------------

def test_compute_auc_roc_uses_predict_proba(trained_model, pca_data, monkeypatch):
    """
    Autograder-proof check:
    - compute_auc_roc must call predict_proba
    - and must NOT use predict() for ROC/AUC
    """
    _, X_test_pca, _, y_test, _ = pca_data

    flags = {"used_proba": False, "used_predict": False}

    original_proba = trained_model.predict_proba
    original_predict = trained_model.predict

    def wrapped_proba(X):
        flags["used_proba"] = True
        return original_proba(X)

    def wrapped_predict(X):
        flags["used_predict"] = True
        return original_predict(X)

    monkeypatch.setattr(trained_model, "predict_proba", wrapped_proba)
    monkeypatch.setattr(trained_model, "predict", wrapped_predict)

    fpr, tpr, auc_score = ds.compute_auc_roc(trained_model, X_test_pca, y_test)

    assert flags["used_proba"], (
        "compute_auc_roc() must use model.predict_proba() to get scores for ROC/AUC."
    )
    assert not flags["used_predict"], (
        "compute_auc_roc() should NOT use model.predict() for ROC/AUC; "
        "ROC requires continuous scores (probabilities)."
    )

    # Basic validity checks
    assert isinstance(fpr, np.ndarray) and isinstance(tpr, np.ndarray)
    assert fpr.shape == tpr.shape
    assert fpr.ndim == 1 and tpr.ndim == 1
    assert np.all((0 <= fpr) & (fpr <= 1))
    assert np.all((0 <= tpr) & (tpr <= 1))
    assert 0.0 <= float(auc_score) <= 1.0


def test_compute_auc_roc_auc_matches_sklearn(trained_model, pca_data):
    """
    Ensure the returned AUC is consistent with sklearn's roc_auc_score
    given the model's probabilities.
    """
    _, X_test_pca, _, y_test, _ = pca_data

    fpr, tpr, auc_score = ds.compute_auc_roc(trained_model, X_test_pca, y_test)

    # Recompute expected AUC from probabilities
    probs = trained_model.predict_proba(X_test_pca)[:, 1]
    expected = roc_auc_score(y_test, probs)

    assert abs(float(auc_score) - float(expected)) < 1e-6, (
        f"AUC mismatch. Returned {auc_score}, expected {expected}."
    )


def test_compute_auc_roc_monotonic(trained_model, pca_data):
    _, X_test_pca, _, y_test, _ = pca_data
    fpr, tpr, _ = ds.compute_auc_roc(trained_model, X_test_pca, y_test)

    # sklearn outputs should be non-decreasing
    assert np.all(np.diff(fpr) >= -1e-10), "FPR should be non-decreasing"
    assert np.all(np.diff(tpr) >= -1e-10), "TPR should be non-decreasing"
