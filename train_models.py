import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from src.data_preprocessing import load_and_clean
from src.feature_engineering import engineer_features, get_feature_target
from sklearn.metrics import adjusted_rand_score
from src import models, evaluation


def main():
    print("Loading data...")
    df = load_and_clean()
    df = engineer_features(df)
    X, y = get_feature_target(df)

    # Target ko 3 classes mein convert kare
    q33, q66 = y.quantile([0.33, 0.66])
    y_class = pd.cut(
        y,
        bins=[-float("inf"), q33, q66, float("inf")],
        labels=["Low", "Medium", "High"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, stratify=y_class, random_state=42
    )
    feature_names = list(X_train.columns)

    os.makedirs("models", exist_ok=True)

    # 1. Supervised models
    print("Training supervised models...")
    sup_models = models.train_supervised_models(X_train, y_train, feature_names)
    sup_metrics = {}
    for name, mdl in sup_models.items():
        res = evaluation.evaluate_classifier(mdl, X_test, y_test)
        sup_metrics[name] = {
            k: float(v)
            for k, v in res.items()
            if k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        }
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(mdl, f"models/{safe_name}.pkl")
        print(f"{name}: {sup_metrics[name]}")

    # 2. Unsupervised
    print("Training K-Means...")
    kmeans = models.train_kmeans(X_train, feature_names)
    clusters = kmeans.predict(X_test)
    ari = adjusted_rand_score(y_test, clusters)
    joblib.dump(kmeans, "models/KMeans.pkl")
    print(f"K-Means ARI: {ari:.3f}")

    # 3. Semi-supervised
    print("Training semi-supervised model...")
    semi, n_label = models.train_semi_supervised(
        X_train, y_train, feature_names, labeled_frac=0.10
    )
    res = evaluation.evaluate_classifier(semi, X_test, y_test)
    semi_metrics = {
        k: float(v)
        for k, v in res.items()
        if k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    }
    joblib.dump(semi, "models/SelfTraining_LR.pkl")
    print(f"Semi-Supervised (10% labels): {semi_metrics}")

    with open("models/metrics.json", "w") as f:
        json.dump(
            {
                "supervised": sup_metrics,
                "semi_supervised": semi_metrics,
                "kmeans_ari": float(ari),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()