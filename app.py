"""Streamlit front end for the churn model. Run with: streamlit run app.py"""
import io

import pandas as pd
import streamlit as st

from src import config, data, forms, model, plots, preprocess

st.set_page_config(page_title="Churn Predictor", layout="wide")


# ---------------------------------------------------------------- data loading

@st.cache_data(show_spinner=False)
def load_bundled(name):
    return data.read_csv(config.DATA_DIR / name)


@st.cache_data(show_spinner=False)
def clean_cached(df, max_missing):
    return data.clean(df, max_missing)


def sidebar():
    st.sidebar.title("Churn Predictor")
    files = data.bundled_files()
    choice = st.sidebar.selectbox("Dataset", files + ["Upload a CSV"])

    if choice == "Upload a CSV":
        upload = st.sidebar.file_uploader("CSV file", type="csv")
        if upload is None:
            st.info("Choose a dataset in the sidebar to begin.")
            st.stop()
        raw = data.read_csv(io.BytesIO(upload.getvalue()))
        label = upload.name
    else:
        raw, label = load_bundled(choice), choice

    max_missing = st.sidebar.slider(
        "Drop columns missing more than", 0.0, 0.5, config.MAX_MISSING_SHARE, 0.01, format="%.2f",
        help="Share of empty cells above which a whole column is discarded.")
    return raw, label, max_missing


# ------------------------------------------------------------------------ tabs

def tab_data(raw, cleaned, notes):
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{len(raw):,}")
    c2.metric("Rows after cleaning", f"{len(cleaned):,}")
    c3.metric("Churn rate", f"{cleaned[config.TARGET].mean():.1%}")

    st.subheader("Cleaning log")
    for note in notes:
        st.write("- " + note)

    st.subheader("Preview")
    st.dataframe(cleaned.head(200), width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(cleaned.describe().T, width="stretch")


def tab_explore(cleaned, X_raw, y):
    X_encoded = preprocess.encode(X_raw)

    left, right = st.columns([1, 2])
    left.pyplot(plots.target_balance(y))
    right.pyplot(plots.target_correlation(X_encoded, y))

    st.subheader("Churn rate by feature")
    column = st.selectbox("Feature", list(X_raw.columns))
    table = data.churn_rate_by(cleaned, column)
    st.pyplot(plots.churn_rate_bars(table, column))

    st.subheader("Redundant features")
    threshold = st.slider("Flag pairs correlated above", 0.5, 0.99, 0.8, 0.01)
    pairs = data.redundant_pairs(X_raw, threshold)
    if pairs.empty:
        st.write("No pairs of numeric features cross that threshold.")
    else:
        st.dataframe(pairs, width="stretch")

    with st.expander("Distributions of numeric features"):
        st.pyplot(plots.histograms(X_raw.select_dtypes("number")))
    with st.expander("Correlation heatmap"):
        st.pyplot(plots.correlation_heatmap(X_encoded))


def tab_train(X_raw, y, ids, dataset_key):
    n_encoded = preprocess.encode(X_raw).shape[1]

    c1, c2, c3, c4 = st.columns(4)
    n_features = c1.slider("Features kept by RFE", 5, n_encoded, min(config.DEFAULT_N_FEATURES, n_encoded))
    test_size = c2.slider("Test share", 0.1, 0.4, config.DEFAULT_TEST_SIZE, 0.05)
    balance = c3.selectbox(
        "Class balancing", model.BALANCE_CHOICES,
        help="Undersample discards majority rows. class_weight keeps all rows and reweights them.")
    folds = c4.slider("CV folds", 3, 10, config.DEFAULT_CV_FOLDS)

    if st.button("Train model", type="primary"):
        settings = model.Settings(n_features, test_size, balance, folds)
        with st.spinner("Fitting model and running cross-validation..."):
            st.session_state["trained"] = model.train(X_raw, y, ids, settings)
            st.session_state["trained_on"] = dataset_key

    trained = active_model(dataset_key)
    if trained is None:
        st.write("Set the options above and press Train model.")
        return

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01,
                          help="Customers scoring at or above this are flagged as likely to churn.")
    result = model.evaluate(trained.y_test, trained.test_proba, threshold)

    m = st.columns(6)
    m[0].metric("Accuracy", f"{result['accuracy']:.3f}")
    m[1].metric("Precision", f"{result['precision']:.3f}")
    m[2].metric("Recall", f"{result['recall']:.3f}")
    m[3].metric("F1", f"{result['f1']:.3f}")
    m[4].metric("ROC AUC", f"{result['roc_auc']:.3f}")
    m[5].metric("Always-retained accuracy", f"{result['majority_baseline']:.3f}",
                help="Accuracy of a model that predicts the majority class for everyone.")

    cv = trained.cv_scores
    st.caption(
        f"Cross-validation on {trained.train_rows:,} training rows: "
        f"accuracy {cv['accuracy'].mean():.3f} (+/- {2 * cv['accuracy'].std():.3f}), "
        f"ROC AUC {cv['roc_auc'].mean():.3f} (+/- {2 * cv['roc_auc'].std():.3f})."
    )

    a, b = st.columns(2)
    a.pyplot(plots.confusion(result["confusion"]))
    b.pyplot(plots.roc(trained.y_test, trained.test_proba, result["roc_auc"]))

    st.subheader("Selected features")
    st.pyplot(plots.coefficients(trained.coefficients))
    st.dataframe(trained.coefficients, width="stretch")


def tab_predict(X_raw, dataset_key):
    trained = active_model(dataset_key)
    if trained is None:
        st.write("Train a model first.")
        return
    row = forms.customer_form(X_raw)
    if row is not None:
        p = float(model.predict_proba(trained, row)[0])
        st.metric("Estimated churn probability", f"{p:.1%}")
        st.write(f"Risk band: **{model.risk_band([p])[0]}**")
        st.caption("Probabilities from an undersampled model run higher than real-world rates, "
                   "so read them as a ranking rather than a literal frequency.")


def tab_export(X_raw, ids, dataset_key):
    trained = active_model(dataset_key)
    if trained is None:
        st.write("Train a model first.")
        return

    scope = st.radio(
        "Customers to score",
        ["Held-out test customers", "Every customer"],
        help="Scoring training customers flatters the model, so the test set is the honest option.")

    if scope == "Held-out test customers":
        table = pd.DataFrame({
            ids.name: trained.test_ids,
            "churn": trained.y_test,
            "churn_probability": trained.test_proba,
        })
    else:
        table = pd.DataFrame({
            ids.name: ids.reset_index(drop=True),
            "churn_probability": model.predict_proba(trained, X_raw),
        })

    table["risk_band"] = model.risk_band(table["churn_probability"])
    table = table.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    st.write(table["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).rename("customers").to_frame().T)
    st.dataframe(table, width="stretch")
    st.download_button("Download CSV", table.to_csv(index=False), "churn_predictions.csv", "text/csv")


def active_model(dataset_key):
    """The trained model, but only if it belongs to the dataset currently on screen."""
    if st.session_state.get("trained_on") == dataset_key:
        return st.session_state.get("trained")
    return None


# ------------------------------------------------------------------------ main

def main():
    raw, label, max_missing = sidebar()
    try:
        data.check_schema(raw)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    cleaned, notes = clean_cached(raw, max_missing)
    ids, X_raw, y = data.split_xy(cleaned)
    dataset_key = (label, max_missing)

    st.title("Customer churn prediction")
    tabs = st.tabs(["Data", "Explore", "Train", "Predict one", "Export scores"])
    with tabs[0]:
        tab_data(raw, cleaned, notes)
    with tabs[1]:
        tab_explore(cleaned, X_raw, y)
    with tabs[2]:
        tab_train(X_raw, y, ids, dataset_key)
    with tabs[3]:
        tab_predict(X_raw, dataset_key)
    with tabs[4]:
        tab_export(X_raw, ids, dataset_key)


main()
