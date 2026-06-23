from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
import numpy as np


def build_preprocessor(feature_names):
    """Numeric columns ke liye median imputation + standard scaling."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_names)], remainder="drop"
    )
    return preprocessor


def make_supervised_pipelines(feature_names):
    """Supervised learning pipelines."""
    preprocessor = build_preprocessor(feature_names)

    return {
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=42,
                        probability=True,
                    ),
                ),
            ]
        ),
        "Neural Network (MLP)": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=400,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def train_supervised_models(X_train, y_train, feature_names):
    """Saare supervised classifiers train kare."""
    pipelines = make_supervised_pipelines(feature_names)
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
    return pipelines


def train_kmeans(X_train, feature_names, n_clusters=3):
    """K-Means unsupervised model train kare."""
    preprocessor = build_preprocessor(feature_names)
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ]
    )
    pipe.fit(X_train)
    return pipe


def train_semi_supervised(X_train, y_train, feature_names, labeled_frac=0.10):
    """Self-Training semi-supervised model with 10% labels."""
    preprocessor = build_preprocessor(feature_names)
    y_train = np.array(y_train)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y_train))
    n_label = int(labeled_frac * len(y_train))

    y_semi = y_train.copy()
    y_semi[idx[n_label:]] = -1

    base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf = SelfTrainingClassifier(
    estimator=base, threshold=0.6, max_iter=100, verbose=0
)

    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_semi)

    return pipe, n_label