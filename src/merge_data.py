"""Task 1: clean, merge and feature-engineer customer data.

Accepts CSV or Excel inputs. The canonical customer key throughout the pipeline
is customer_id_legacy; customer_id_new (for example A178) is normalized only
for matching social profiles to that legacy key.

Run from the repository root:
    python src/merge_data.py \
        --social data/customer_social_profiles.xlsx \
        --transactions data/customer_transactions.xlsx
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SENTIMENT_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}


def read_table(path):
    """Read CSV or Excel based on the file extension."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_legacy_id(values):
    """Normalize both 178 and A178 to the integer legacy ID 178."""
    extracted = values.astype("string").str.strip().str.extract(r"(\d+)$")[0]
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def load_and_clean_social(path):
    df = read_table(path)
    required = {"customer_id_new", "social_media_platform", "engagement_score",
                "purchase_interest_score", "review_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Social profile file is missing columns: {sorted(missing)}")
    print(f"[social] raw shape: {df.shape}")
    print(f"[social] exact duplicate rows: {df.duplicated().sum()}")
    df = df.drop_duplicates().copy()
    df["customer_id_legacy"] = normalize_legacy_id(df["customer_id_new"])
    df = df.dropna(subset=["customer_id_legacy"]).copy()
    df["customer_id_legacy"] = df["customer_id_legacy"].astype(int)
    df["sentiment_score"] = df["review_sentiment"].map(SENTIMENT_MAP)
    print(f"[social] after dedup: {df.shape}, unique legacy customers: "
          f"{df['customer_id_legacy'].nunique()}")
    return df


def aggregate_social(df):
    """One row per legacy customer; prevent a many-to-many transaction join."""
    g = df.groupby("customer_id_legacy")
    return g.agg(
        n_social_records=("customer_id_legacy", "size"),
        avg_engagement=("engagement_score", "mean"),
        max_engagement=("engagement_score", "max"),
        avg_purchase_interest=("purchase_interest_score", "mean"),
        avg_sentiment=("sentiment_score", "mean"),
        n_platforms=("social_media_platform", "nunique"),
        top_platform=("social_media_platform", lambda s: s.mode().iloc[0]),
    ).reset_index()


def load_and_clean_transactions(path):
    df = read_table(path)
    required = {"customer_id_legacy", "transaction_id", "purchase_amount",
                "purchase_date", "product_category", "customer_rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Transactions file is missing columns: {sorted(missing)}")
    print(f"[txn] raw shape: {df.shape}")
    df = df.drop_duplicates().copy()
    df["customer_id_legacy"] = pd.to_numeric(df["customer_id_legacy"], errors="raise").astype(int)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df["rating_missing"] = df["customer_rating"].isna().astype(int)
    df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
    return df


def merge(txn, social_agg):
    before = len(txn)
    merged = txn.merge(social_agg, on="customer_id_legacy", how="left",
                       validate="m:1", indicator=True)
    assert len(merged) == before, "merge changed the number of transactions"
    matched = int((merged["_merge"] == "both").sum())
    print(f"[merge] transactions: {before}, matched social profiles: {matched}, "
          f"unmatched: {before - matched}")
    unmatched_ids = sorted(merged.loc[merged["_merge"] == "left_only",
                                      "customer_id_legacy"].unique().tolist())
    print(f"[merge] legacy customer IDs with no social profile: {unmatched_ids}")
    merged["has_social_profile"] = (merged["_merge"] == "both").astype(int)
    return merged.drop(columns="_merge")


def engineer(df):
    df = df.copy()
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.dayofweek
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
    df["amount_bucket"] = pd.cut(
        df["purchase_amount"], [0, 150, 300, 600],
        labels=["low", "mid", "high"]
    ).astype("string").fillna("Unknown")
    social_num = ["avg_engagement", "max_engagement", "avg_purchase_interest",
                  "avg_sentiment", "n_social_records", "n_platforms"]
    for c in social_num:
        df[c] = df[c].fillna(df[c].median())
    df["top_platform"] = df["top_platform"].fillna("Unknown")
    df["engagement_x_interest"] = df["avg_engagement"] * df["avg_purchase_interest"]
    return df


def make_plots(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["purchase_amount"].dropna(), bins=20, edgecolor="black")
    ax.set(title="Distribution of purchase amount", xlabel="Purchase amount", ylabel="Transactions")
    fig.tight_layout()
    fig.savefig(outdir / "purchase_amount_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    categories = list(df["product_category"].dropna().unique())
    values = [df.loc[df["product_category"] == category, "purchase_amount"].dropna()
              for category in categories]
    ax.boxplot(values, tick_labels=categories)
    ax.set(title="Purchase amount by product category", xlabel="Product category",
           ylabel="Purchase amount")
    fig.tight_layout()
    fig.savefig(outdir / "purchase_amount_outliers.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--social", default="data/customer_social_profiles.xlsx")
    ap.add_argument("--transactions", default="data/customer_transactions.xlsx")
    ap.add_argument("--out", default="data/merged_dataset.csv")
    ap.add_argument("--plots", default="reports")
    a = ap.parse_args()

    social = aggregate_social(load_and_clean_social(a.social))
    txn = load_and_clean_transactions(a.transactions)
    merged = engineer(merge(txn, social))
    print(f"\\n[summary] merged shape: {merged.shape}")
    print(f"[summary] null values: {int(merged.isna().sum().sum())}")
    print("[summary] target counts:\\n", merged["product_category"].value_counts().to_string())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out, index=False)
    make_plots(merged, Path(a.plots))
    print(f"\\nSaved {a.out} with {len(merged)} transaction rows")


if __name__ == "__main__":
    main()
