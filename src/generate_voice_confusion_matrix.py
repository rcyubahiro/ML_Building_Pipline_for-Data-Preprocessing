
"""Generate a current voice confusion matrix using grouped cross-validation."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

from common import candidate_models, _grouped_splits


def main():
    data_path = Path("data/audio_features.csv")
    output_path = Path("reports/task3_voice_confusion_matrices.png")

    metadata_columns = ["member", "phrase", "source_file", "augmentation"]

    df = pd.read_csv(data_path)

    missing = [col for col in metadata_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_columns = [
        col for col in df.columns if col not in metadata_columns
    ]

    X = df[feature_columns].apply(pd.to_numeric, errors="raise")
    y = df["member"].astype(str)
    groups = df["source_file"].astype(str)

    classes = sorted(y.unique())

    # Use the same grouped validation procedure as train_voice.py.
    splits = _grouped_splits(
        X, y, groups, max_splits=5, seed=42
    )

    # Use the selected Random Forest model.
    model_factory = candidate_models()["random_forest"]

    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predictions = model.predict(X.iloc[test_idx])

        all_true.extend(y.iloc[test_idx].tolist())
        all_pred.extend(predictions.tolist())

        print(
            f"Fold {fold}: "
            f"{len(train_idx)} training rows, "
            f"{len(test_idx)} validation rows"
        )

    accuracy = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(
        all_true, all_pred, average="macro"
    )

    print("\n=== Current Voice Cross-Validation Results ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1:  {macro_f1:.3f}")

    # Generate confusion matrix from combined out-of-fold predictions.
    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay.from_predictions(
        all_true,
        all_pred,
        labels=classes,
        display_labels=classes,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(
        "Voice Recognition Confusion Matrix\n"
        "Random Forest - Grouped by Original Recording"
    )
    ax.set_xlabel("Predicted Member")
    ax.set_ylabel("Actual Member")

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved current confusion matrix: {output_path}")


if __name__ == "__main__":
    main()
