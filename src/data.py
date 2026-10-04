"""Reading, cleaning and basic summaries of the raw churn exports."""
import numpy as np
import pandas as pd

from . import config


def bundled_files():
    return sorted(p.name for p in config.DATA_DIR.glob("*.csv"))


def read_csv(source):
    """`source` can be a path or a file-like object (e.g. a Streamlit upload)."""
    return pd.read_csv(source)


def find_id_column(df):
    for name in config.ID_COLUMNS:
        if name in df.columns:
            return name
    return None


def check_schema(df):
    if config.TARGET not in df.columns:
        raise ValueError(f"The file needs a '{config.TARGET}' column with 0/1 labels.")
    labels = set(df[config.TARGET].dropna().unique())
    if not labels <= {0, 1}:
        raise ValueError(f"'{config.TARGET}' must contain only 0 and 1, found {sorted(labels)}.")


def clean(df, max_missing=config.MAX_MISSING_SHARE):
    """Return a cleaned copy of `df` plus a list of notes describing what changed."""
    notes = []
    id_col = find_id_column(df)

    out = df.drop_duplicates()
    if id_col:
        out = out.drop_duplicates(subset=id_col)
    removed = len(df) - len(out)
    if removed:
        notes.append(
            f"Removed {removed:,} duplicate rows. Left in, copies of the same customer "
            "end up in both the training and the test set and inflate every metric."
        )

    feature_cols = [c for c in out.columns if c not in (config.TARGET, id_col)]
    sparse = [c for c in feature_cols if out[c].isna().mean() > max_missing]
    if sparse:
        out = out.drop(columns=sparse)
        notes.append(f"Dropped columns with more than {max_missing:.0%} missing values: {', '.join(sparse)}.")

    if "credit_score" in out.columns:
        invalid = out["credit_score"] < config.MIN_VALID_CREDIT_SCORE
        if invalid.any():
            out = out[~invalid]
            notes.append(f"Removed {int(invalid.sum()):,} rows with a credit score below {config.MIN_VALID_CREDIT_SCORE}.")

    before = len(out)
    out = out.dropna()
    if len(out) < before:
        notes.append(f"Removed {before - len(out):,} rows that still had missing values.")

    notes.append(f"{len(out):,} rows and {out.shape[1]} columns remain.")
    return out.reset_index(drop=True), notes


def split_xy(df):
    """Separate identifiers, raw features and the target."""
    id_col = find_id_column(df)
    ids = df[id_col] if id_col else pd.Series(df.index, name="row")
    y = df[config.TARGET].astype(int)
    drop = [c for c in (config.TARGET, id_col) if c]
    return ids, df.drop(columns=drop), y


def redundant_pairs(X, threshold=0.8):
    """Pairs of numeric columns whose absolute correlation is at least `threshold`."""
    corr = X.select_dtypes("number").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "correlation"]
    pairs = pairs[pairs["correlation"] >= threshold]
    return pairs.sort_values("correlation", ascending=False).reset_index(drop=True)


def churn_rate_by(df, column, bins=5):
    """Churn rate and head-count per value of `column` (numeric columns are binned)."""
    series = df[column]
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        series = pd.qcut(series, q=bins, duplicates="drop")
    grouped = df.groupby(series.astype(str))[config.TARGET].agg(["mean", "size"])
    grouped.columns = ["churn_rate", "customers"]
    grouped.index.name = "group"
    return grouped.reset_index()
