"""Command-line simulation of the flow in the assignment diagram:

    Start -> Face recognition -> Product model -> Voice validation -> Display predicted product
                  |fail                                |fail
            Access Denied                         Access Denied

Usage (from the repo root):
    python src/app.py --demo                       # authorized + two unauthorized attempts
    python src/app.py --face X.jpg --voice Y.wav   # one transaction
    python src/app.py                              # interactive prompts
"""
import argparse
import sys
import warnings
from pathlib import Path

import cv2
import joblib
import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from audio_pipeline import SR, extract_features as audio_features  # noqa: E402
from image_pipeline import crop_face, extract_features as image_features  # noqa: E402
from common import open_set_distance  # noqa: E402
from train_product import build_X  # noqa: E402

MODELS = Path("models")


def load_bundle(name):
    p = MODELS / name
    if not p.exists():
        sys.exit(f"Missing {p}. Run the train_*.py scripts first.")
    return joblib.load(p)


def top_prediction(bundle, feats):
    """Returns (best member, probability, distance to that member's centroid)."""
    X = pd.DataFrame([feats]).reindex(columns=bundle["features"], fill_value=0.0)
    proba = bundle["model"].predict_proba(X)[0]
    i = int(np.argmax(proba))
    member = bundle["model"].classes_[i]
    dist = open_set_distance(bundle["open_set"], X.iloc[0].values, member)
    return member, float(proba[i]), dist


def accepted(conf, dist, bundle):
    return conf >= bundle["threshold"] and dist <= bundle["open_set"]["radius"]


def recognise_face(path, bundle):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return top_prediction(bundle, image_features(crop_face(img)))


def verify_voice(path, bundle):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return top_prediction(bundle, audio_features(y))


def recommend(customer_id, bundle, merged):
    rows = merged[merged["customer_id"] == customer_id].sort_values("purchase_date")
    if rows.empty:
        raise ValueError(f"customer {customer_id} not in merged dataset")
    X = build_X(rows.tail(1)).reindex(columns=bundle["features"], fill_value=0)
    proba = bundle["model"].predict_proba(X)[0]
    order = np.argsort(proba)[::-1]
    return [(bundle["model"].classes_[i], float(proba[i])) for i in order]


def run_transaction(face_path, voice_path, face_b, voice_b, prod_b, merged, cmap, label=""):
    print("\n" + "=" * 64)
    print(f"TRANSACTION {label}".strip())
    print(f"  face image : {face_path}\n  voice clip : {voice_path}")
    print("-" * 64)

    member, conf, dist = recognise_face(face_path, face_b)
    print(f"[1] Face recognition: closest member '{member}' (confidence {conf:.2f} "
          f"vs {face_b['threshold']:.2f}; distance {dist:.2f} vs limit {face_b['open_set']['radius']:.2f})")
    if not accepted(conf, dist, face_b):
        print("    -> ACCESS DENIED: face not recognised.")
        return False
    print(f"    -> Face accepted as {member}.")

    if member not in cmap:
        print(f"    -> ACCESS DENIED: no customer account linked to '{member}'.")
        return False
    cid = cmap[member]
    ranking = recommend(cid, prod_b, merged)
    print(f"[2] Product model ran for customer {cid} (result held until voice is verified).")

    speaker, vconf, vdist = verify_voice(voice_path, voice_b)
    print(f"[3] Voice validation: closest speaker '{speaker}' (confidence {vconf:.2f} "
          f"vs {voice_b['threshold']:.2f}; distance {vdist:.2f} vs limit {voice_b['open_set']['radius']:.2f})")
    if not accepted(vconf, vdist, voice_b) or speaker != member:
        why = ("voice does not match the recognised face" if speaker != member
               else "voice not recognised")
        print(f"    -> ACCESS DENIED: {why}.")
        return False
    print("    -> Voice approved.")

    print(f"[4] PREDICTED PRODUCT for {member}: {ranking[0][0]} (p={ranking[0][1]:.2f})")
    print("    Other likely categories: " + ", ".join(f"{c} ({p:.2f})" for c, p in ranking[1:3]))
    return True


def first_file(folder, exts):
    fs = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in exts)
    return fs[0] if fs else None


def demo(args, face_b, voice_b, prod_b, merged, cmap):
    member = face_b["classes"][0]
    good_face = Path(args.images) / f"{member}_smiling.jpg"
    good_voice = Path(args.audio) / f"{member}_yes_approve.wav"
    bad_face = first_file(Path(args.images) / "unauthorized", {".jpg", ".jpeg", ".png"})
    bad_voice = first_file(Path(args.audio) / "unauthorized", {".wav", ".mp3", ".m4a", ".flac"})
    results = [run_transaction(good_face, good_voice, face_b, voice_b, prod_b, merged, cmap,
                               "1: authorized user")]
    if bad_face:
        results.append(run_transaction(bad_face, good_voice, face_b, voice_b, prod_b, merged, cmap,
                                       "2: UNAUTHORIZED face"))
    if bad_voice:
        results.append(run_transaction(good_face, bad_voice, face_b, voice_b, prod_b, merged, cmap,
                                       "3: authorized face, UNAUTHORIZED voice"))
    print("\n" + "=" * 64)
    print(f"Demo finished: {sum(results)} approved, {len(results) - sum(results)} denied.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face"); ap.add_argument("--voice")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--images", default="images"); ap.add_argument("--audio", default="audio")
    ap.add_argument("--merged", default="data/merged_dataset.csv")
    ap.add_argument("--map", default="data/member_customer_map.csv")
    args = ap.parse_args()

    face_b, voice_b, prod_b = (load_bundle(n) for n in
                               ("face_model.pkl", "voice_model.pkl", "product_model.pkl"))
    merged = pd.read_csv(args.merged, parse_dates=["purchase_date"])
    cmap = dict(pd.read_csv(args.map).values)

    if args.demo:
        demo(args, face_b, voice_b, prod_b, merged, cmap)
    elif args.face and args.voice:
        run_transaction(args.face, args.voice, face_b, voice_b, prod_b, merged, cmap)
    else:
        print("Product prediction system (type 'q' to quit)")
        while True:
            f = input("\nFace image path: ").strip()
            if f.lower() == "q":
                break
            v = input("Voice clip path: ").strip()
            if v.lower() == "q":
                break
            try:
                run_transaction(f, v, face_b, voice_b, prod_b, merged, cmap)
            except Exception as e:  # keep the loop alive for the demo
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
