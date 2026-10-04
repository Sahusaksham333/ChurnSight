"""Matplotlib figures. Each function returns a Figure so the caller decides how to show it."""
import math

import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve


BLUE = "#3F5D7D"
RED = "#B5483C"
LABELS = ["Retained", "Churned"]


def target_balance(y):
    fig = Figure(figsize=(4, 3))
    ax = fig.subplots()
    counts = y.value_counts().sort_index()
    ax.bar(LABELS, counts.values, color=[BLUE, RED])
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}\n({value / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Customers by outcome")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def histograms(df, ncols=4):
    cols = list(df.columns)
    nrows = math.ceil(len(cols) / ncols)
    fig = Figure(figsize=(3.2 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
    for ax, col in zip(axes, cols):
        ax.hist(df[col], bins=min(df[col].nunique(), 30), color=BLUE)
        ax.set_title(col, fontsize=9)
        ax.set_yticks([])
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def churn_rate_bars(table, column):
    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    ax.bar(table["group"], table["churn_rate"], color=BLUE)
    for i, (rate, n) in enumerate(zip(table["churn_rate"], table["customers"])):
        ax.text(i, rate, f"{rate:.0%}\nn={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, min(1.0, table["churn_rate"].max() * 1.3 + 0.05))
    ax.set_ylabel("Churn rate")
    ax.set_title(f"Churn rate by {column}")
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def target_correlation(X_encoded, y, top=25):
    corr = X_encoded.corrwith(y).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)[:top].iloc[::-1]
    fig = Figure(figsize=(7, 0.28 * len(corr) + 1))
    ax = fig.subplots()
    ax.barh(corr.index, corr.values, color=[RED if v > 0 else BLUE for v in corr.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation with churn")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def correlation_heatmap(X_encoded):
    corr = X_encoded.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig = Figure(figsize=(11, 9))
    ax = fig.subplots()
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0,
                vmin=-0.6, vmax=0.6, square=True, linewidths=0.4, cbar_kws={"shrink": 0.6}, ax=ax)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def confusion(cm):
    fig = Figure(figsize=(4.2, 3.4))
    ax = fig.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def roc(y_true, proba, auc):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig = Figure(figsize=(4.2, 3.4))
    ax = fig.subplots()
    ax.plot(fpr, tpr, color=BLUE, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def coefficients(table):
    table = table.iloc[::-1]
    fig = Figure(figsize=(7, 0.3 * len(table) + 1))
    ax = fig.subplots()
    ax.barh(table["feature"], table["coefficient"],
            color=[RED if v > 0 else BLUE for v in table["coefficient"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Model coefficients (red raises churn risk, blue lowers it)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
