"""Shared helpers for the three training scripts."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METRICS_PATH = Path("reports/metrics.json")


def candidate_models():
    """Two model families, as the brief suggests (Logistic Regression, Random Forest)."""
    return {
        "logistic_regression": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0)),
        ]),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42),
    }


def score(y_true, y_pred, proba, classes):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
    }


def cross_validate(make_model, X, y, groups=None, n_splits=5, seed=42):
    """Grouped CV when `groups` is given (all augmented copies of one source stay together)."""
    classes = np.array(sorted(set(y)))
    if groups is not None:
        splitter = GroupKFold(n_splits=min(n_splits, len(set(groups))))
        splits = splitter.split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
    y_true, y_pred, proba = [], [], []
    for tr, te in splits:
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        p = np.zeros((len(te), len(classes)))
        idx = [list(classes).index(c) for c in m.classes_]
        p[:, idx] = m.predict_proba(X.iloc[te])
        p = p / p.sum(axis=1, keepdims=True)
        y_true += list(y.iloc[te]); y_pred += list(m.predict(X.iloc[te])); proba.append(p)
    return score(y_true, y_pred, np.vstack(proba), list(classes))


def pick_best(X, y, groups=None):
    results = {name: cross_validate(f, X, y, groups) for name, f in candidate_models().items()}
    best = max(results, key=lambda k: (results[k]["f1_macro"], -results[k]["log_loss"]))
    return best, results


def _rms_dist(Z, centroid):
    return np.sqrt(((Z - centroid) ** 2).mean(axis=1))


def fit_open_set(X, y, groups, margin=1.1):
    """Reject strangers: a closed-set classifier always names *some* member, so we also require
    the sample to lie close to that member's centroid (standardised feature space, RMS distance).
    The radius is calibrated on held-out groups (expression / phrase), i.e. on genuinely new
    samples of known people, then widened by `margin`."""
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    y = np.asarray(y); groups = np.asarray(groups)
    held_out = []
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        for m in sorted(set(y[te])):
            if (tr & (y == m)).any():
                c = Z[tr & (y == m)].mean(axis=0)
                held_out += list(_rms_dist(Z[te & (y == m)], c))
    centroids = {m: Z[y == m].mean(axis=0) for m in sorted(set(y))}
    return {"mean": scaler.mean_, "scale": scaler.scale_, "centroids": centroids,
            "radius": float(max(held_out) * margin)}


def open_set_distance(open_set, x_row, member):
    z = (np.asarray(x_row, dtype=float) - open_set["mean"]) / open_set["scale"]
    return float(_rms_dist(z[None, :], open_set["centroids"][member])[0])


def save_metrics(key, payload):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    allm = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    allm[key] = payload
    METRICS_PATH.write_text(json.dumps(allm, indent=2))


def print_results(title, best, results):
    print(f"\n=== {title} ===")
    for name, r in results.items():
        mark = "  <- selected" if name == best else ""
        print(f"{name:20s} acc={r['accuracy']:.3f}  f1_macro={r['f1_macro']:.3f}  "
              f"log_loss={r['log_loss']:.3f}{mark}")
