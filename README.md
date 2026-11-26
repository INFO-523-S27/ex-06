# Ex-06: Classifying Spam Emails with Decision Trees

## Objective

In this exercise, you will build and evaluate a **binary classification model** to predict whether an email is **spam or not spam** using a **Decision Tree classifier**.

Beyond simply fitting a model, this assignment emphasizes:

* Understanding **classification evaluation metrics**
* Visualizing **decision boundaries**
* Reasoning about the effects of **dimensionality reduction (PCA)** on model behavior

By the end of this exercise, you should be able to explain *how* a model makes decisions—not just whether it performs well.

---

## Prerequisites

You should have:

* Python 3.9+
* GitHub access
* Familiarity with:

  * pandas, numpy
  * scikit-learn classifiers and metrics
  * train/test splits
  * basic plotting with matplotlib

Required libraries (installed via `requirements.txt`):

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

The dataset **`spam.csv`** is provided in the `data/` directory.

> ⚠️ You do **not** need to create or submit a Jupyter notebook for this exercise.

---

## Dataset

The data comes from **Vincent Arel-Bundock’s Rdatasets collection**:

[https://vincentarelbundock.github.io/Rdatasets/index.html](https://vincentarelbundock.github.io/Rdatasets/index.html)

We use the **Spam Email dataset**, originally collected at **Hewlett-Packard Labs** and later distributed via the **UCI Machine Learning Repository**.

* 4,601 emails
* Binary outcome: spam vs. not spam
* Numeric features describing word and character frequencies

More details:

* Dataset docs: [https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/spam7.html](https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/spam7.html)
* Original UCI dataset: [https://archive.ics.uci.edu/dataset/94/spambase](https://archive.ics.uci.edu/dataset/94/spambase)

---

## Metadata

| Variable  | Type      | Description                                            |
| --------- | --------- | ------------------------------------------------------ |
| `crl.tot` | double    | Total length of uninterrupted capital letter sequences |
| `dollar`  | double    | `%` of characters that are `$`                         |
| `bang`    | double    | `%` of characters that are `!`                         |
| `money`   | double    | `%` of words that are `"money"`                        |
| `n000`    | double    | `%` of words that are `"000"`                          |
| `make`    | double    | `%` of words that are `"make"`                         |
| `yesno`   | character | Target: `n` = not spam, `y` = spam                     |

(Source: [TidyTuesday](https://github.com/rfordatascience/tidytuesday/blob/master/data/2023/2023-08-15/readme.md))

---

## Research Question

> **Can we predict whether an email is spam using a decision tree classifier?**

More importantly:

* What does the decision boundary look like?
* Which evaluation metric best reflects model performance?
* How does PCA affect interpretability and performance?

---

## Step 1: Setup and Data Preprocessing

You will:

1. Load the dataset
2. Encode categorical variables
3. Define features and target
4. Split into training and testing sets
5. Apply **PCA** to reduce features to **two components**

Example workflow (provided for reference):

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

spam = pd.read_csv("data/spam.csv")

categorical_columns = spam.select_dtypes(include=["object", "category"]).columns
for col in categorical_columns:
    spam[col] = LabelEncoder().fit_transform(spam[col])

X = spam.drop("yesno", axis=1)
y = spam["yesno"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

> 💡 **Why PCA here?**
> Decision trees do *not* require dimensionality reduction—but PCA allows us to **visualize** the decision boundary in 2D.

---

## Step 2: Model Training & Decision Boundary Visualization

You will:

* Train a **Decision Tree classifier** on the PCA-transformed data
* Implement a function called **`decisionplot()`** that:

  * Creates a 2D mesh grid
  * Uses the trained model to predict across the grid
  * Visualizes the decision boundary and training points

Example usage:

```python
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_pca, y_train)

decisionplot(
    dtree,
    pd.DataFrame(X_train_pca, columns=["PC1", "PC2"]),
    y_train
)
```

> ⚠️ Your `decisionplot()` must evaluate predictions over a **grid**, not just the training points.

---

## Step 3: Model Evaluation

Evaluate your model using:

* Accuracy
* Precision
* Recall
* F1 score
* **AUC-ROC**

Example metrics (provided for reference):

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

predictions = dtree.predict(X_test_pca)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted")
recall = recall_score(y_test, predictions, average="weighted")
f1 = f1_score(y_test, predictions, average="weighted")
```

You must additionally:

* Compute **ROC curve**
* Compute **AUC** using `predict_proba()` (not `predict()`)

---

## What You Must Implement

In `ds.py`, you are responsible for implementing:

1. **`decisionplot()`**

   * Generates a decision boundary visualization using a mesh grid
2. **AUC-ROC calculation**

   * Uses model probabilities
   * Returns valid ROC components and AUC score

Your code must pass the provided **autograder tests**.

---

## Discussion Prompts (Answer in Discussion via D2L)

Consider:

* Which metric is *most informative* for spam detection? Why?
* How does the decision boundary help explain model behavior?
* What is gained—and lost—by applying PCA before training?

There is no single “correct” answer, but your reasoning should be grounded in the results.

---

## Submission Instructions

* Submit **only** your completed `ds.py` file via GitHub
* Do **not** submit notebooks or additional scripts
* Ensure all tests pass before submitting

> The autograder checks **correctness**, **methodology**, and **code quality**.
