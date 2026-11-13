"""Task 3 – train the voice verification model.

Reads data/audio_features.csv, evaluates with a leave-one-phrase-out split
(train on one phrase, test on the other, both directions), then fits the
final model on all rows and saves models/voice_model.pkl plus
reports/voice_model_metrics.json.

Why leave-one-phrase-out: augmented copies of one recording are near
duplicates. A random row split would put copies of the same clip in train
and test and inflate the scores. Holding out a whole phrase tests whether
the model recognises the *speaker* on words it has not heard.

Run from the repo root:  python src/train_voice.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_pipeline import (FEATURES_CSV, FEATURE_COLUMNS, PHRASES, PROJECT_ROOT, SEED,
                            UNAUTHORIZED_DIR, list_recordings, verify_voice)

MODEL_PATH = PROJECT_ROOT / "models" / "voice_model.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "voice_model_metrics.json"
HYPERPARAMS = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "random_state": SEED}


def make_model():
    return Pipeline([("scaler", StandardScaler()),
                     ("classifier", RandomForestClassifier(**HYPERPARAMS))])


def evaluate(df):
    """Train on one phrase, test on the other; returns per-fold results."""
    folds = []
    labels = sorted(df["member"].unique())
    for test_phrase in PHRASES:
        train, test = df[df.phrase != test_phrase], df[df.phrase == test_phrase]
        if train.empty or test.empty:
            continue
        model = make_model().fit(train[FEATURE_COLUMNS], train["member"])
        pred = model.predict(test[FEATURE_COLUMNS])
        proba = model.predict_proba(test[FEATURE_COLUMNS])
        folds.append({
            "train_phrase": train.phrase.iloc[0], "test_phrase": test_phrase,
            "train_rows": len(train), "test_rows": len(test),
            "accuracy": float(accuracy_score(test.member, pred)),
            "f1_macro": float(f1_score(test.member, pred, average="macro", zero_division=0)),
            "log_loss": float(log_loss(test.member, proba, labels=model.classes_)),
            "genuine_confidence_min": float(proba.max(axis=1).min()),
            "confusion_matrix": confusion_matrix(test.member, pred, labels=labels).tolist(),
            "labels": labels,
            "y_true": test.member.tolist(), "y_pred": pred.tolist(),
        })
    return folds


def main():
    df = pd.read_csv(FEATURES_CSV)
    folds = evaluate(df)
    summary = {k: float(np.mean([f[k] for f in folds])) for k in ("accuracy", "f1_macro", "log_loss")}

    final_model = make_model().fit(df[FEATURE_COLUMNS], df["member"])

    # Confidence threshold for accepting a voice: the lowest top-class
    # probability the model gave a *genuine* held-out clip, so known members
    # are still accepted. Outsider clips are reported against it.
    threshold = round(min(f["genuine_confidence_min"] for f in folds), 2) if folds else 0.5
    outsiders = []
    if UNAUTHORIZED_DIR.exists():
        for path in list_recordings(UNAUTHORIZED_DIR):
            outsiders.append({"file": path.name, **verify_voice(path, final_model, threshold)})

    MODEL_PATH.parent.mkdir(exist_ok=True); METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    metrics = {
        "model": "voice_verification", "algorithm": "RandomForestClassifier (StandardScaler first)",
        "hyperparameters": HYPERPARAMS, "split": "leave-one-phrase-out (train on one phrase, test on the other)",
        "members": sorted(df.member.unique()), "rows": len(df), "features": len(FEATURE_COLUMNS),
        "mean_over_folds": summary,
        "folds": [{k: v for k, v in f.items() if k not in ("y_true", "y_pred")} for f in folds],
        "acceptance_threshold": threshold, "unauthorized_checks": outsiders,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("members", "rows", "mean_over_folds",
                                               "acceptance_threshold", "unauthorized_checks")}, indent=2))
    return metrics, folds, final_model


if __name__ == "__main__":
    main()
