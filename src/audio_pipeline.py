"""Task 3: load voice samples, plot waveform + spectrogram, augment, extract features -> audio_features.csv.

Expected naming:
    audio/<member>_<phrase>.wav
    phrase in {yes_approve, confirm_transaction}
e.g. audio/chriss_yes_approve.wav

Run from the repo root:
    python src/audio_pipeline.py --audio audio --out data/audio_features.csv
"""
import argparse
import re
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SR = 16000
PHRASES = ("yes_approve", "confirm_transaction")
N_MFCC = 13
RNG = np.random.default_rng(42)


def parse_name(path):
    m = re.match(r"^(?P<member>.+?)_(?P<phrase>yes_approve|confirm_transaction)$", path.stem, re.I)
    return (m["member"].lower(), m["phrase"].lower()) if m else None


def load_audio(folder):
    items, skipped = [], []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
            continue
        parsed = parse_name(p)
        if parsed is None:
            skipped.append(p.name)
            continue
        y, _ = librosa.load(p, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)  # cut leading/trailing silence
        if len(y) < SR * 0.2:
            skipped.append(p.name)
            continue
        items.append({"member": parsed[0], "phrase": parsed[1], "path": p, "y": y})
    if skipped:
        print(f"[warn] skipped (bad name / too short): {skipped}")
    return items


# ---------- augmentations ----------
def pitch_shift(y):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=2)


def time_stretch(y):
    return librosa.effects.time_stretch(y, rate=1.1)


def add_noise(y):
    noise = RNG.normal(0, 0.005 * max(np.abs(y).max(), 1e-6), size=y.shape)
    return y + noise


AUGMENTATIONS = {
    "original": lambda y: y,
    "pitch_shift_+2": pitch_shift,
    "time_stretch_1.1": time_stretch,
    "background_noise": add_noise,
}


# ---------- features ----------
def extract_features(y):
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc_{i + 1}_std"] = float(mfcc[i].std())
    feats["spectral_rolloff_mean"] = float(librosa.feature.spectral_rolloff(y=y, sr=SR).mean())
    feats["spectral_centroid_mean"] = float(librosa.feature.spectral_centroid(y=y, sr=SR).mean())
    feats["zero_crossing_rate_mean"] = float(librosa.feature.zero_crossing_rate(y).mean())
    feats["rms_energy_mean"] = float(librosa.feature.rms(y=y).mean())
    feats["duration_s"] = float(len(y) / SR)
    return feats


def plot_wave_and_spec(item, out_png):
    y = item["y"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5))
    librosa.display.waveshow(y, sr=SR, ax=a1)
    a1.set(title=f"Waveform: {item['member']} - {item['phrase']}", xlabel="Time (s)", ylabel="Amplitude")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="hz", ax=a2)
    a2.set(title="Spectrogram")
    fig.colorbar(img, ax=a2, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    matplotlib.use("Agg")  # script mode: write PNGs, no window (notebooks keep their own backend)
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="audio")
    ap.add_argument("--out", default="data/audio_features.csv")
    ap.add_argument("--plots", default="reports/figures")
    a = ap.parse_args()

    items = load_audio(a.audio)
    if not items:
        raise SystemExit(f"No valid audio in {a.audio}. Expected <member>_<phrase>.wav")
    members = sorted({i["member"] for i in items})
    print(f"Loaded {len(items)} recordings for {len(members)} members: {members}")
    for m in members:
        miss = set(PHRASES) - {i["phrase"] for i in items if i["member"] == m}
        if miss:
            print(f"[warn] {m} is missing phrases: {sorted(miss)}")

    Path(a.plots).mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        plot_wave_and_spec(it, Path(a.plots) / f"audio_{it['member']}_{it['phrase']}.png")
        for aug_name, fn in AUGMENTATIONS.items():
            row = {"member": it["member"], "phrase": it["phrase"],
                   "source_file": it["path"].name, "augmentation": aug_name}
            row.update(extract_features(fn(it["y"])))
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {a.out}: {df.shape[0]} rows x {df.shape[1]} cols "
          f"({len(AUGMENTATIONS)} variants per recording)")


if __name__ == "__main__":
    main()
