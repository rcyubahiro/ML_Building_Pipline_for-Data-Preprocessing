"""Facial recognition model: image_features.csv -> who is this member?

Validation: grouped CV where each fold holds out ONE EXPRESSION (e.g. train on neutral + smiling,
test on surprised, including its augmented copies). That tests generalisation to a new photo of the
same person instead of memorising near-duplicates.

    python src/train_face.py
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import candidate_models, fit_open_set, pick_best, print_results, save_metrics

META = ["member", "expression", "source_file", "augmentation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/image_features.csv")
    ap.add_argument("--out", default="models/face_model.pkl")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="minimum predicted probability to accept a face as a known member")
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    X, y, groups = df.drop(columns=META), df["member"], df["expression"]
    print(f"{len(df)} rows, {X.shape[1]} features, members={sorted(y.unique())}")

    best, results = pick_best(X, y, groups)
    print_results("Face model (leave-one-expression-out CV)", best, results)

    model = candidate_models()[best]().fit(X, y)
    open_set = fit_open_set(X, y, groups)
    print(f"Open-set rejection radius (RMS z-distance): {open_set['radius']:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns), "threshold": a.threshold,
                 "classes": list(model.classes_), "name": best, "open_set": open_set}, a.out)
    save_metrics("face_model", {"selected": best, "cv": results, "threshold": a.threshold,
                                "open_set_radius": open_set["radius"]})
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
