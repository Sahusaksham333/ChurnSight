"""Turning raw feature tables into something a linear model can digest."""
import numpy as np
import pandas as pd

from . import config


def categorical_columns(X):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def encode(X):
    """One-hot encode text columns.

    The "unknown" level of each column is dropped, so it acts as the reference
    category and we avoid the dummy variable trap.
    """
    out = X.copy()
    cats = categorical_columns(out)
    if cats:
        out[cats] = out[cats].fillna(config.UNKNOWN_LABEL)
        out = pd.get_dummies(out, columns=cats, dtype=int)
    unknown = [c for c in out.columns if c.endswith(f"_{config.UNKNOWN_LABEL}")]
    return out.drop(columns=unknown)


def align(X_encoded, columns):
    """Force a frame to have exactly the training columns, filling gaps with zero."""
    return X_encoded.reindex(columns=columns, fill_value=0)


def undersample(X, y, seed=config.RANDOM_STATE):
    """Shrink every class to the size of the smallest one, sampling without replacement."""
    rng = np.random.default_rng(seed)
    labels = y.to_numpy()
    n_keep = y.value_counts().min()
    chosen = [
        rng.choice(np.flatnonzero(labels == label), size=n_keep, replace=False)
        for label in np.unique(labels)
    ]
    keep = rng.permutation(np.concatenate(chosen))
    return X.iloc[keep], y.iloc[keep]
