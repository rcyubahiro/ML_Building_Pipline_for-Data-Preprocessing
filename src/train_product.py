"""Product recommendation model: merged_dataset.csv -> product_category.

    python src/train_product.py
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from common import candidate_models, pick_best, print_results, save_metrics, score

DROP = ["customer_id", "transaction_id", "purchase_date", "amount_bucket", "product_category"]
TARGET = "product_category"


def build_X(df):
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    return pd.get_dummies(X, columns=["top_platform"], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/merged_dataset.csv")
    ap.add_argument("--out", default="models/product_model.pkl")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    X, y = build_X(df), df[TARGET]
    print(f"{len(df)} rows, {X.shape[1]} features; classes={y.value_counts().to_dict()}")

    # Hold out 20% for a final test; choose the model on the other 80% with 5-fold CV
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    best, cv = pick_best(Xtr.reset_index(drop=True), ytr.reset_index(drop=True))
    print_results("Product model (5-fold stratified CV on 80% train)", best, cv)

    model = candidate_models()[best]().fit(Xtr, ytr)
    test = score(yte, model.predict(Xte), model.predict_proba(Xte), list(model.classes_))
    chance = float(y.value_counts(normalize=True).max())
    print(f"\nHold-out test ({len(yte)} rows): acc={test['accuracy']:.3f}  "
          f"f1_macro={test['f1_macro']:.3f}  log_loss={test['log_loss']:.3f}  "
          f"(majority-class baseline acc={chance:.3f})")

    final = candidate_models()[best]().fit(X, y)  # refit on everything for the app
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": list(X.columns), "classes": list(final.classes_),
                 "name": best}, a.out)
    save_metrics("product_model", {"selected": best, "cv": cv, "holdout_test": test,
                                   "majority_baseline_accuracy": chance})
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
