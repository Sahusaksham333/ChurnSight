"""Training, validation and scoring of the churn classifier."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .preprocess import align, encode, undersample

BALANCE_CHOICES = ("undersample", "class_weight", "none")


@dataclass
class Settings:
    n_features: int = config.DEFAULT_N_FEATURES
    test_size: float = config.DEFAULT_TEST_SIZE
    balance: str = "undersample"
    cv_folds: int = config.DEFAULT_CV_FOLDS


@dataclass
class TrainedModel:
    pipeline: Pipeline
    columns: list
    coefficients: pd.DataFrame
    cv_scores: dict
    test_ids: pd.Series
    y_test: pd.Series
    test_proba: np.ndarray
    train_rows: int
    settings: Settings = field(default_factory=Settings)


def build_pipeline(n_features, n_available, class_weight=None):
    """Scaler, optional RFE selection, then the final logistic regression."""
    def logit():
        return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=config.RANDOM_STATE)

    steps = [("scale", StandardScaler())]
    if n_features < n_available:
        steps.append(("select", RFE(logit(), n_features_to_select=n_features, step=2)))
    steps.append(("clf", logit()))
    return Pipeline(steps)


def train(X_raw, y, ids, settings):
    X = encode(X_raw)
    X_train, X_test, y_train, y_test, _, ids_test = train_test_split(
        X, y, ids,
        test_size=settings.test_size,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    if settings.balance == "undersample":
        X_train, y_train = undersample(X_train, y_train)
    class_weight = "balanced" if settings.balance == "class_weight" else None

    n_features = min(settings.n_features, X.shape[1])
    pipeline = build_pipeline(n_features, X.shape[1], class_weight)

    # The scaler and RFE sit inside the pipeline, so they are refit on each
    # training fold and never see the validation fold.
    folds = StratifiedKFold(settings.cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    cv = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=["accuracy", "roc_auc"])

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    return TrainedModel(
        pipeline=pipeline,
        columns=list(X.columns),
        coefficients=_coefficient_table(pipeline, X.columns),
        cv_scores={"accuracy": cv["test_accuracy"], "roc_auc": cv["test_roc_auc"]},
        test_ids=ids_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        test_proba=proba,
        train_rows=len(X_train),
        settings=settings,
    )


def _coefficient_table(pipeline, columns):
    names = np.asarray(columns)
    if "select" in pipeline.named_steps:
        names = names[pipeline.named_steps["select"].get_support()]
    coef = pipeline.named_steps["clf"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coef})
    # Features are standardised, so this is the change in odds per one standard deviation.
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def evaluate(y_true, proba, threshold=0.5):
    pred = (np.asarray(proba) >= threshold).astype(int)
    majority = max(y_true.mean(), 1 - y_true.mean())
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "majority_baseline": majority,
        "confusion": confusion_matrix(y_true, pred, labels=[0, 1]),
    }


def predict_proba(model, X_raw):
    X = align(encode(X_raw), model.columns)
    return model.pipeline.predict_proba(X)[:, 1]


def risk_band(proba):
    low, high = config.RISK_BANDS
    return pd.cut(proba, bins=[-0.001, low, high, 1.0], labels=["Low", "Medium", "High"])
