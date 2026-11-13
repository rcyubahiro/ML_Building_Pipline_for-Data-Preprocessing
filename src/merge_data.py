"""Task 1: clean, merge and feature-engineer customer_social_profiles + customer_transactions.

Run from the repo root:
    python src/merge_data.py \
        --social data/customer_social_profiles.csv \
        --transactions data/customer_transactions.csv
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SENTIMENT_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}


def load_and_clean_social(path):
    df = pd.read_csv(path)
    print(f"[social] raw shape: {df.shape}")
    print(f"[social] nulls:\n{df.isna().sum().to_string()}")
    n_dup = df.duplicated().sum()
    print(f"[social] exact duplicate rows: {n_dup}")
    df = df.drop_duplicates().copy()

    # Harmonise the key: 'A178' -> 178 so it matches customer_id_legacy (int)
    df["customer_id"] = df["customer_id_new"].str.extract(r"(\d+)")[0].astype(int)
    df["sentiment_score"] = df["review_sentiment"].map(SENTIMENT_MAP)
    print(f"[social] after dedup: {df.shape}, unique customers: {df['customer_id'].nunique()}")
    return df


def aggregate_social(df):
    """One row per customer so the join cannot multiply transaction rows."""
    g = df.groupby("customer_id")
    agg = g.agg(
        n_social_records=("customer_id", "size"),
        avg_engagement=("engagement_score", "mean"),
        max_engagement=("engagement_score", "max"),
        avg_purchase_interest=("purchase_interest_score", "mean"),
        avg_sentiment=("sentiment_score", "mean"),
        n_platforms=("social_media_platform", "nunique"),
        top_platform=("social_media_platform", lambda s: s.mode().iloc[0]),
    ).reset_index()
    return agg


def load_and_clean_transactions(path):
    df = pd.read_csv(path, parse_dates=["purchase_date"])
    print(f"[txn] raw shape: {df.shape}")
    print(f"[txn] nulls:\n{df.isna().sum().to_string()}")
    print(f"[txn] duplicate transaction_id: {df['transaction_id'].duplicated().sum()}")
    df = df.rename(columns={"customer_id_legacy": "customer_id"})
    df["rating_missing"] = df["customer_rating"].isna().astype(int)
    df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
    return df


def merge(txn, social_agg):
    before = len(txn)
    merged = txn.merge(social_agg, on="customer_id", how="left", validate="m:1", indicator=True)
    assert len(merged) == before, "merge changed the number of transactions"
    matched = (merged["_merge"] == "both").sum()
    print(f"[merge] transactions: {before}, matched to a social profile: {matched}, "
          f"unmatched: {before - matched}")
    unmatched_ids = sorted(merged.loc[merged["_merge"] == "left_only", "customer_id"].unique())
    print(f"[merge] customer IDs with no social profile: {unmatched_ids}")
    merged["has_social_profile"] = (merged["_merge"] == "both").astype(int)
    return merged.drop(columns="_merge")


def engineer(df):
    df = df.copy()
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.dayofweek
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
    df["amount_bucket"] = pd.cut(df["purchase_amount"], [0, 150, 300, 600],
                                 labels=["low", "mid", "high"]).astype(str)

    # Customers without a social profile: impute with column medians + keep the flag
    social_num = ["avg_engagement", "max_engagement", "avg_purchase_interest",
                  "avg_sentiment", "n_social_records", "n_platforms"]
    for c in social_num:
        df[c] = df[c].fillna(df[c].median())
    df["top_platform"] = df["top_platform"].fillna("Unknown")
    df["engagement_x_interest"] = df["avg_engagement"] * df["avg_purchase_interest"]
    return df


def make_plots(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["purchase_amount"], bins=20, kde=True, ax=ax)
    ax.set(title="Distribution of purchase amount", xlabel="Purchase amount", ylabel="Transactions")
    fig.tight_layout(); fig.savefig(outdir / "eda_purchase_amount_dist.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="product_category", y="purchase_amount", ax=ax)
    ax.set(title="Purchase amount by product category (outliers shown as points)",
           xlabel="Product category", ylabel="Purchase amount")
    fig.tight_layout(); fig.savefig(outdir / "eda_amount_by_category_box.png", dpi=150); plt.close(fig)

    num = ["purchase_amount", "customer_rating", "avg_engagement", "avg_purchase_interest",
           "avg_sentiment", "n_social_records", "engagement_x_interest"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation of numeric features")
    fig.tight_layout(); fig.savefig(outdir / "eda_correlation_heatmap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    order = df["product_category"].value_counts().index
    sns.countplot(data=df, x="product_category", order=order, ax=ax)
    ax.set(title="Target balance: product category", xlabel="Product category", ylabel="Transactions")
    fig.tight_layout(); fig.savefig(outdir / "eda_target_balance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="top_platform", hue="product_category", ax=ax)
    ax.set(title="Product category by customer's top social platform",
           xlabel="Top platform", ylabel="Transactions")
    fig.tight_layout(); fig.savefig(outdir / "eda_platform_vs_category.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--social", required=True)
    ap.add_argument("--transactions", required=True)
    ap.add_argument("--out", default="data/merged_dataset.csv")
    ap.add_argument("--plots", default="reports/figures")
    a = ap.parse_args()

    social = load_and_clean_social(a.social)
    social_agg = aggregate_social(social)
    txn = load_and_clean_transactions(a.transactions)
    merged = engineer(merge(txn, social_agg))

    print("\n[summary] dtypes:\n", merged.dtypes.to_string())
    print("\n[summary] describe:\n", merged.describe().T.round(2).to_string())
    print("\n[summary] target counts:\n", merged["product_category"].value_counts().to_string())
    print("\n[summary] nulls after cleaning:", int(merged.isna().sum().sum()))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out, index=False)
    make_plots(merged, Path(a.plots))
    print(f"\nSaved {a.out} {merged.shape} and plots to {a.plots}")


if __name__ == "__main__":
    main()
