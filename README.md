# Customer churn predictor

A Streamlit app that cleans a customer-activity export, explores it, trains a
logistic regression churn model with recursive feature elimination, and scores
customers. It is a rebuild of the original `Project 2 - EDA.py` and
`Project 2 - Model.py` scripts.

## Run it

    pip install -r requirements.txt
    streamlit run app.py

Run the tests with `python -m pytest`.

## Layout

    app.py              page layout and tab logic only
    src/config.py       constants (target name, seeds, thresholds)
    src/data.py         loading, cleaning, summaries
    src/preprocess.py   one-hot encoding, column alignment, undersampling
    src/model.py        pipeline, training, cross-validation, metrics, scoring
    src/plots.py        matplotlib figures
    src/forms.py        single-customer input form
    data/               both CSV exports and the column description
    tests/              pytest checks for the core logic

## Differences from the original scripts

- Duplicate customers are removed before splitting. The processed file held
  thousands of repeated rows, so the same customer could land in both the
  training and test sets.
- Sparse columns are dropped before rows are filtered. The original
  `credit_score >= 300` filter silently discarded every row where the score
  was missing, then dropped the column anyway.
- Undersampling draws without replacement. `np.random.choice` defaults to
  sampling with replacement, which duplicated majority-class rows.
- The scaler and RFE live inside a scikit-learn pipeline, so cross-validation
  refits them on each fold instead of reusing statistics from the whole
  training set.
- ROC AUC and a majority-class baseline are reported next to accuracy.
- The decision threshold is adjustable after training.
- Pie charts are replaced by churn-rate bars, which answer the question the
  EDA was actually asking.
- `np.bool` (removed from NumPy) is gone, and the "SVM Accuracy" print label
  that described a logistic regression is fixed.

## Data note

`new_churn_data.csv` uses the column names from the description file
(`userid`, `rent_or_own`, `payfreq`, ...), while `churn_data.csv` uses the
older names the original scripts expected (`user`, `housing`, ...). The app
detects the identifier and text columns automatically, so either file works,
as does any upload with a binary `churn` column.
# ChurnSight
