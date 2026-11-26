"""
Ex-06: Classifying Spam Emails with Decision Trees

This module contains helper functions for:
- Loading and preprocessing the spam dataset
- Applying PCA
- Training a decision tree classifier
- Visualizing decision boundaries
- Evaluating classification performance

Students are expected to COMPLETE the TODO sections.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# ---------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------

def load_spam_data(path="data/spam.csv"):
    """
    Load the spam dataset.

    Parameters
    ----------
    path : str
        Path to spam.csv

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def encode_categoricals(df):
    """
    Encode categorical columns using LabelEncoder.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Encoded dataframe
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def split_features_target(df, target="yesno"):
    """
    Split dataframe into features and target.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# ---------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------

def apply_pca(X_train, X_test, n_components=2):
    """
    Apply PCA to training and test data.

    Returns
    -------
    X_train_pca : np.ndarray
    X_test_pca : np.ndarray
    pca : PCA object
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca


# ---------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------

def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a Decision Tree classifier.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def decisionplot(model, X, y, resolution=0.02):
    """
    Plot decision boundary for a 2D classifier.

    Parameters
    ----------
    model : trained classifier
    X : pd.DataFrame with columns ['PC1', 'PC2']
    y : array-like target
    resolution : float
        Grid resolution

    TODO:
    - Create a meshgrid
    - Predict class labels over the grid
    - Plot decision boundary and data points
    """
    raise NotImplementedError("Implement decisionplot using the steps above.")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_classification(model, X_test, y_test):
    """
    Compute standard classification metrics.

    Returns
    -------
    dict
    """
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted"),
        "recall": recall_score(y_test, preds, average="weighted"),
        "f1": f1_score(y_test, preds, average="weighted"),
    }

    return metrics


def compute_auc_roc(model, X_test, y_test):
    """
    Compute ROC curve and AUC score.

    TODO:
    - Use predict_proba
    - Compute fpr, tpr
    - Return AUC
    """
    raise NotImplementedError("Compute the ROC curve and AUC using model probabilities.")
