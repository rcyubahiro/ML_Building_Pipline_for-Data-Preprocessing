"""Shared helpers for the three training scripts."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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


def _grouped_splits(X, y, groups, max_splits=5, seed=42):
    """Create group-aware, class-stratified folds.

    Every augmented row from a source recording stays in one fold. At least two
    independent source recordings per member are needed for this validation.
    """
    y_array = np.asarray(y)
    group_array = np.asarray(groups)
    per_class_groups = [
        len(set(group_array[y_array == label])) for label in np.unique(y_array)
    ]
    n_splits = min(max_splits, min(per_class_groups))
    if n_splits < 2:
        raise ValueError(
            "Grouped voice validation needs at least two distinct source_file "
            "recordings for every member. Collect more original recordings."
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return splitter.split(X, y, group_array)


def cross_validate(make_model, X, y, groups=None, n_splits=5, seed=42):
    """Evaluate without allowing augmented copies of one source into both folds."""
    classes = np.array(sorted(set(y)))
    if groups is not None:
        splits = _grouped_splits(X, y, groups, max_splits=n_splits, seed=seed)
    else:
        splits = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        ).split(X, y)

    y_true, y_pred, probabilities = [], [], []
    for train_idx, test_idx in splits:
        model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_probabilities = np.zeros((len(test_idx), len(classes)))
        class_positions = [list(classes).index(c) for c in model.classes_]
        fold_probabilities[:, class_positions] = model.predict_proba(X.iloc[test_idx])
        fold_probabilities /= fold_probabilities.sum(axis=1, keepdims=True)

        y_true.extend(y.iloc[test_idx].tolist())
        y_pred.extend(model.predict(X.iloc[test_idx]).tolist())
        probabilities.append(fold_probabilities)

    return score(y_true, y_pred, np.vstack(probabilities), list(classes))


def pick_best(X, y, groups=None):
    results = {
        name: cross_validate(factory, X, y, groups)
        for name, factory in candidate_models().items()
    }
    best = max(
        results,
        key=lambda name: (results[name]["f1_macro"], -results[name]["log_loss"]),
    )
    return best, results


def _rms_dist(Z, centroid):
    return np.sqrt(((Z - centroid) ** 2).mean(axis=1))


def fit_open_set(X, y, groups, unknown_X=None, unknown_groups=None, min_unknown_sources=5):
    """Fit centroids and calibrate a conservative distance rejection radius.

    Genuine distances are estimated by leaving out complete source recordings.
    If independent outsider recordings are supplied, the radius is placed
    between the 95th percentile of genuine distances and the 5th percentile of
    nearest-centroid outsider distances, but only when those distributions
    are separated. Without outsider data (or when they overlap), the result is
    explicitly marked as NOT calibrated for unknown-speaker rejection.
    """
    y_array = np.asarray(y)
    group_array = np.asarray(groups)

    genuine_distances = []
    for group in sorted(set(group_array)):
        train_mask = group_array != group
        test_mask = group_array == group
        if not train_mask.any():
            continue

        # Fit scaling on the training recordings only for this held-out fold.
        fold_scaler = StandardScaler().fit(X.loc[train_mask])
        z_train = fold_scaler.transform(X.loc[train_mask])
        z_test = fold_scaler.transform(X.loc[test_mask])
        y_train = y_array[train_mask]
        y_test = y_array[test_mask]

        for member in sorted(set(y_test)):
            member_train = z_train[y_train == member]
            if len(member_train):
                centroid = member_train.mean(axis=0)
                genuine_distances.extend(
                    _rms_dist(z_test[y_test == member], centroid).tolist()
                )

    if not genuine_distances:
        raise ValueError("Could not calculate genuine held-out voice distances.")

    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    centroids = {
        member: Z[y_array == member].mean(axis=0)
        for member in sorted(set(y_array))
    }

    # Genuine-only radius is useful as a diagnostic, but is not proof that
    # unknown speakers will be rejected.
    genuine_limit = float(np.quantile(genuine_distances, 0.95))

    unknown_calibrated = False
    unknown_source_count = 0
    unknown_min_distance = None
    radius = genuine_limit
    calibration_note = "No outsider recordings supplied; unknown-speaker rejection is uncalibrated."

    if unknown_X is not None and len(unknown_X):
        if list(unknown_X.columns) != list(X.columns):
            raise ValueError(
                "Unknown feature columns must match the training feature columns "
                "in both names and order."
            )

        unknown_Z = scaler.transform(unknown_X)
        centroid_matrix = np.vstack([centroids[m] for m in centroids])
        # Distance to the closest known speaker is the conservative impostor score.
        unknown_distances = np.column_stack([
            _rms_dist(unknown_Z, centroid) for centroid in centroid_matrix
        ]).min(axis=1)

        if unknown_groups is None:
            # One row must represent one independent, original outsider recording.
            unknown_source_count = len(unknown_X)
        else:
            unknown_groups = np.asarray(unknown_groups)
            if len(unknown_groups) != len(unknown_X):
                raise ValueError("unknown_groups must have one value per outsider feature row.")
            unknown_source_count = len(set(unknown_groups))
        unknown_min_distance = float(np.min(unknown_distances))
        unknown_limit = float(np.quantile(unknown_distances, 0.05))

        if unknown_source_count >= min_unknown_sources and genuine_limit < unknown_limit:
            radius = float((genuine_limit + unknown_limit) / 2.0)
            unknown_calibrated = True
            calibration_note = (
                "Genuine and outsider distance distributions were separated on "
                "the supplied calibration samples. Validate on additional unseen speakers."
            )
        else:
            calibration_note = (
                "Outsider data was insufficient or overlapped genuine distances; "
                "unknown-speaker rejection remains uncalibrated."
            )

    return {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "centroids": centroids,
        "radius": radius,
        "genuine_radius_95": genuine_limit,
        "unknown_calibrated": unknown_calibrated,
        "unknown_source_count": unknown_source_count,
        "unknown_min_distance": unknown_min_distance,
        "calibration_note": calibration_note,
    }


def open_set_distance(open_set, x_row, member):
    z = (np.asarray(x_row, dtype=float) - open_set["mean"]) / open_set["scale"]
    return float(_rms_dist(z[None, :], open_set["centroids"][member])[0])


def save_metrics(key, payload):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    all_metrics[key] = payload
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))


def print_results(title, best, results):
    print(f"\n=== {title} ===")
    for name, result in results.items():
        mark = "  <- selected" if name == best else ""
        print(
            f"{name:20s} acc={result['accuracy']:.3f}  "
            f"f1_macro={result['f1_macro']:.3f}  "
            f"log_loss={result['log_loss']:.3f}{mark}"
        )
