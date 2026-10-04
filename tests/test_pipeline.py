import numpy as np
import pandas as pd

from src import data, model, preprocess


def toy_frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "userid": np.arange(n),
        "churn": (rng.random(n) < 0.3).astype(int),
        "age": rng.integers(18, 70, n).astype(float),
        "visits": rng.poisson(5, n),
        "payfreq": rng.choice(["Weekly", "Monthly", "na"], n),
    })
    df["visits"] += df["churn"] * 3
    return df


def test_clean_removes_duplicate_customers():
    df = toy_frame()
    doubled = pd.concat([df, df.head(50)], ignore_index=True)
    cleaned, _ = data.clean(doubled)
    assert len(cleaned) == len(df)


def test_encode_drops_unknown_level():
    X = toy_frame().drop(columns=["userid", "churn"])
    encoded = preprocess.encode(X)
    assert "payfreq_na" not in encoded.columns
    assert {"payfreq_Weekly", "payfreq_Monthly"} <= set(encoded.columns)


def test_undersample_balances_classes_without_repeats():
    df = toy_frame()
    X, y = df[["age", "visits"]], df["churn"]
    X_bal, y_bal = preprocess.undersample(X, y)
    assert y_bal.value_counts().nunique() == 1
    assert not X_bal.index.duplicated().any()


def test_training_beats_chance_and_scoring_matches_columns():
    df = toy_frame()
    ids, X, y = data.split_xy(df)
    trained = model.train(X, y, ids, model.Settings(n_features=3, cv_folds=3))
    assert model.evaluate(trained.y_test, trained.test_proba)["roc_auc"] > 0.6
    probs = model.predict_proba(trained, X.head(5))
    assert len(probs) == 5 and ((probs >= 0) & (probs <= 1)).all()
