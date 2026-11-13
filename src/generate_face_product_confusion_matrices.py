
"""Generate current face and product confusion matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from common import candidate_models, pick_best, _grouped_splits
from train_product import build_X, TARGET


REPORTS = Path("reports")


def save_matrix(y_true, y_pred, labels, title, filename):
    REPORTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=labels,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    fig.tight_layout()
    output = REPORTS / filename
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


def generate_face_matrix():
    print("\n========== FACE MODEL ==========")

    df = pd.read_csv("data/image_features.csv")

    meta = ["member", "expression", "source_file", "augmentation"]
    X = df.drop(columns=meta)
    y = df["member"].astype(str)
    groups = df["expression"].astype(str)

    best, results = pick_best(X, y, groups)
    print(f"Selected model: {best}")

    if best != "random_forest":
        raise RuntimeError(
            "The selected face model differs from the expected Random Forest. "
            "Review the current metrics before proceeding."
        )

    predictions_true = []
    predictions_pred = []

    splits = _grouped_splits(
        X, y, groups, max_splits=5, seed=42
    )

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = candidate_models()[best]()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predicted = model.predict(X.iloc[test_idx])

        predictions_true.extend(y.iloc[test_idx].tolist())
        predictions_pred.extend(predicted.tolist())

        print(
            f"Face fold {fold}: "
            f"{len(train_idx)} training rows, "
            f"{len(test_idx)} validation rows"
        )

    accuracy = accuracy_score(predictions_true, predictions_pred)
    macro_f1 = f1_score(
        predictions_true, predictions_pred, average="macro"
    )

    print(f"Face accuracy: {accuracy:.3f}")
    print(f"Face macro F1: {macro_f1:.3f}")

    labels = sorted(y.unique())

    save_matrix(
        predictions_true,
        predictions_pred,
        labels,
        "Face Recognition Confusion Matrix\n"
        "Random Forest - Grouped by Expression",
        "task3_face_confusion_matrix.png",
    )


def generate_product_matrix():
    print("\n========== PRODUCT MODEL ==========")

    df = pd.read_csv("data/merged_dataset.csv")

    X = build_X(df)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    best, results = pick_best(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
    )

    print(f"Selected model: {best}")

    if best != "random_forest":
        raise RuntimeError(
            "The selected product model differs from the expected Random Forest. "
            "Review the current metrics before proceeding."
        )

    model = candidate_models()[best]()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    print(f"Product test rows: {len(y_test)}")
    print(f"Product accuracy: {accuracy:.3f}")
    print(f"Product macro F1: {macro_f1:.3f}")

    labels = list(model.classes_)

    save_matrix(
        y_test,
        predictions,
        labels,
        "Product Recommendation Confusion Matrix\n"
        "Random Forest - Hold-out Test Set",
        "task3_product_confusion_matrix.png",
    )


if __name__ == "__main__":
    generate_face_matrix()
    generate_product_matrix()
