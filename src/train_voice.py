"""Train the voiceprint classifier from audio_features.csv.

Validation groups by source_file, so augmented copies of one original recording
cannot leak between training and validation folds.

Optional outsider calibration CSV:
    --unknown-features data/unknown_audio_features.csv

The outsider CSV must contain the same numeric feature columns, in the same
order, and one row per independently recorded outsider sample. It must not
contain augmented copies counted as independent recordings.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import (
    candidate_models,
    fit_open_set,
    pick_best,
    print_results,
    save_metrics,
    open_set_distance,
)

META = ["member", "phrase", "source_file", "augmentation"]
MIN_OUTSIDER_SOURCES_FOR_CALIBRATION = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/audio_features.csv")
    parser.add_argument("--unknown-features", default=None,
                        help="Optional CSV of outsider feature rows using the same feature columns.")
    parser.add_argument("--out", default="models/voice_model.pkl")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum predicted class probability; not an outsider-calibrated threshold.")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    missing = [column for column in META if column not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing required metadata columns: {missing}")

    feature_columns = [column for column in df.columns if column not in META]
    if not feature_columns:
        raise ValueError("No numeric audio feature columns were found.")

    X = df[feature_columns].apply(pd.to_numeric, errors="raise")
    y = df["member"].astype(str)
    # Critical: group by original recording, NOT by phrase.
    groups = df["source_file"].astype(str)

    source_counts = df.assign(_member=y, _source=groups).groupby("_member")["_source"].nunique()
    print(f"{len(df)} rows, {X.shape[1]} features, members={sorted(y.unique())}")
    print("Independent original recordings per member:")
    for member, count in source_counts.items():
        print(f"  {member}: {count}")

    best, results = pick_best(X, y, groups)
    print_results("Voice model (grouped by original source recording)", best, results)

    unknown_df = None
    unknown_X = None
    unknown_groups = None
    calibration_unknown_X = None
    calibration_unknown_groups = None
    if args.unknown_features:
        unknown_df = pd.read_csv(args.unknown_features)
        if "source_file" not in unknown_df.columns:
            raise ValueError(
                "Outsider feature CSV must include source_file so independent recordings "
                "can be counted correctly."
            )
        missing_unknown = [column for column in feature_columns if column not in unknown_df.columns]
        if missing_unknown:
            raise ValueError(f"Outsider CSV is missing feature columns: {missing_unknown}")
        unknown_X = unknown_df[feature_columns].apply(pd.to_numeric, errors="raise")
        unknown_groups = unknown_df["source_file"].astype(str)
        outsider_count = int(unknown_groups.nunique())
        print(
            f"Loaded {len(unknown_X)} outsider rows from "
            f"{outsider_count} independent recordings."
        )
        if len(unknown_X) != outsider_count:
            raise ValueError(
                "The outsider CSV must have exactly one row per original recording; "
                "do not include augmented copies."
            )
        if outsider_count >= MIN_OUTSIDER_SOURCES_FOR_CALIBRATION:
            calibration_unknown_X = unknown_X
            calibration_unknown_groups = unknown_groups
            print("[info] Outsider set meets the minimum source-count gate for "
                  "experimental calibration. This is not a guarantee of production-grade security.")
        else:
            print(
                f"[warning] Only {outsider_count} independent outsider recording(s). "
                f"At least {MIN_OUTSIDER_SOURCES_FOR_CALIBRATION} are required by this "
                "script before they can influence the rejection radius. They will be "
                "reported as evaluation samples only; unknown-speaker rejection remains "
                "uncalibrated."
            )

    open_set = fit_open_set(
        X, y, groups,
        unknown_X=calibration_unknown_X,
        unknown_groups=calibration_unknown_groups,
    )
    print(f"Genuine 95th-percentile distance: {open_set['genuine_radius_95']:.3f}")
    print(f"Stored rejection radius: {open_set['radius']:.3f}")
    print(f"Outsider rejection calibrated: {open_set['unknown_calibrated']}")
    print(open_set["calibration_note"])

    model = candidate_models()[best]().fit(X, y)

    outsider_evaluation = []
    if unknown_X is not None:
        print("\\n=== Outsider sample evaluation (not a calibration claim) ===")
        for row_index, row in unknown_X.iterrows():
            probabilities = model.predict_proba(pd.DataFrame([row], columns=feature_columns))[0]
            best_index = int(probabilities.argmax())
            predicted_member = str(model.classes_[best_index])
            distance = float(open_set_distance(open_set, row.to_numpy(dtype=float), predicted_member))
            result = {
                "source_file": str(unknown_groups.iloc[row_index]),
                "predicted_member": predicted_member,
                "confidence": float(probabilities[best_index]),
                "distance": distance,
                "within_stored_radius": bool(distance <= float(open_set["radius"])),
            }
            outsider_evaluation.append(result)
            print(
                f"{result['source_file']}: predicted={predicted_member}, "
                f"confidence={result['confidence']:.3f}, distance={distance:.3f}, "
                f"within_radius={result['within_stored_radius']}"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "threshold": args.threshold,
            "classes": list(model.classes_),
            "name": best,
            "open_set": open_set,
        },
        args.out,
    )

    if outsider_evaluation:
        evaluation_path = Path("reports/voice_outsider_evaluation.csv")
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(outsider_evaluation).to_csv(evaluation_path, index=False)
        print(f"Saved outsider evaluation: {evaluation_path}")

    save_metrics(
        "voice_model",
        {
            "selected": best,
            "cv": results,
            "validation_group": "source_file",
            "threshold": args.threshold,
            "open_set_radius": open_set["radius"],
            "genuine_radius_95": open_set["genuine_radius_95"],
            "unknown_calibrated": open_set["unknown_calibrated"],
            "unknown_source_count": open_set["unknown_source_count"],
            "outsider_evaluation_count": len(outsider_evaluation),
            "calibration_note": open_set["calibration_note"],
        },
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
