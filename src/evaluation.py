import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    adjusted_rand_score,
)


def evaluate_classifier(model, X_test, y_test):
    """Classifier evaluate kare aur metrics return kare."""
    y_pred = model.predict(X_test)
    labels = sorted(y_test.unique())

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "y_pred": y_pred,
    }


def classification_report_df(model, X_test, y_test):
    """Classification report as DataFrame."""
    y_pred = model.predict(X_test)
    return pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()