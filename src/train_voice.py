"""Voiceprint verification model: audio_features.csv -> which member is speaking?

Validation: grouped CV where each fold holds out ONE PHRASE (train on 'yes approve', test on
'confirm transaction' and vice versa), including augmented copies of the held-out recordings.

    python src/train_voice.py
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import candidate_models, fit_open_set, pick_best, print_results, save_metrics

META = ["member", "phrase", "source_file", "augmentation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/audio_features.csv")
    ap.add_argument("--out", default="models/voice_model.pkl")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="minimum predicted probability to accept a voice")
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    X, y, groups = df.drop(columns=META), df["member"], df["phrase"]
    print(f"{len(df)} rows, {X.shape[1]} features, members={sorted(y.unique())}")

    best, results = pick_best(X, y, groups)
    print_results("Voice model (leave-one-phrase-out CV)", best, results)

    model = candidate_models()[best]().fit(X, y)
    open_set = fit_open_set(X, y, groups)
    print(f"Open-set rejection radius (RMS z-distance): {open_set['radius']:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns), "threshold": a.threshold,
                 "classes": list(model.classes_), "name": best, "open_set": open_set}, a.out)
    save_metrics("voice_model", {"selected": best, "cv": results, "threshold": a.threshold,
                                 "open_set_radius": open_set["radius"]})
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
