"""Widgets for scoring a single customer by hand."""
import pandas as pd
import streamlit as st

from . import config


def _is_flag(series):
    return set(series.dropna().unique()) <= {0, 1}


def customer_form(X_raw, columns_per_row=3):
    """Render one input per raw feature. Returns a one-row frame after submit, else None."""
    values = {}
    with st.form("single_customer"):
        cols = st.columns(columns_per_row)
        for i, name in enumerate(X_raw.columns):
            series = X_raw[name]
            box = cols[i % columns_per_row]

            if not pd.api.types.is_numeric_dtype(series):
                options = sorted(series.astype(str).unique())
                default = options.index(config.UNKNOWN_LABEL) if config.UNKNOWN_LABEL in options else 0
                values[name] = box.selectbox(name, options, index=default)
            elif _is_flag(series):
                values[name] = int(box.checkbox(name, value=bool(series.median())))
            elif pd.api.types.is_integer_dtype(series):
                values[name] = box.number_input(
                    name, min_value=int(series.min()), max_value=int(series.max()),
                    value=int(series.median()), step=1)
            else:
                values[name] = box.number_input(
                    name, min_value=float(series.min()), max_value=float(series.max()),
                    value=float(series.median()))
        submitted = st.form_submit_button("Estimate churn risk")
    return pd.DataFrame([values]) if submitted else None
