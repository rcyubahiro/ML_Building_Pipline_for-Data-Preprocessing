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



def load_unknown_audio(folder):
    """Load outsider clips without assigning them a registered member identity."""
    supported = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Outsider audio folder does not exist: {folder}")

    items = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in supported:
            continue
        y, _ = librosa.load(p, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) < SR * 0.2:
            print(f"[warn] skipped outsider clip (too short): {p.name}")
            continue
        items.append({"source_file": p.name, "path": p, "y": y})
    return items


# ---------- augmentations ----------
def pitch_up(y):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=2)


def pitch_down(y):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=-2)


def speed_up(y):
    return librosa.effects.time_stretch(y, rate=1.1)


def slow_down(y):
    return librosa.effects.time_stretch(y, rate=0.9)


def add_noise(y):
    # Approximately 20 dB signal-to-noise ratio.
    signal_rms = float(np.sqrt(np.mean(np.square(y))) + 1e-9)
    noise_rms = signal_rms / (10 ** (20 / 20))
    return y + RNG.normal(0, noise_rms, size=y.shape)


def volume_down(y):
    return 0.5 * y  # approximately -6 dB


def time_shift(y):
    # Add a 0.2-second leading delay without wrapping the end of the clip.
    delay = int(0.2 * SR)
    return np.concatenate((np.zeros(delay, dtype=y.dtype), y))


AUGMENTATIONS = {
    "original": lambda y: y,
    "pitch_up": pitch_up,
    "pitch_down": pitch_down,
    "speed_up": speed_up,
    "slow_down": slow_down,
    "noise": add_noise,
    "volume_down": volume_down,
    "time_shift": time_shift,
}

# ---------- features ----------
def extract_features(y):
    """Return the same 32 numeric columns used by the checked-in voice feature table."""
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(mfcc[i].mean())
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_std"] = float(mfcc[i].std())

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85)
    rms = librosa.feature.rms(y=y)
    f0, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=SR, frame_length=2048, hop_length=512
    )
    voiced_f0 = f0[np.isfinite(f0)]
    feats["rolloff_mean"] = float(rolloff.mean())
    feats["rolloff_std"] = float(rolloff.std())
    feats["energy_mean"] = float(rms.mean())
    feats["energy_std"] = float(rms.std())
    feats["pitch_mean"] = float(voiced_f0.mean()) if len(voiced_f0) else 0.0
    feats["pitch_std"] = float(voiced_f0.std()) if len(voiced_f0) else 0.0
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
    ap.add_argument("--unknown-audio", default=None,
                    help="Optional folder of outsider recordings; no identity labels or augmentation.")
    ap.add_argument("--unknown-out", default="data/unknown_audio_features.csv")
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

    # Outsider samples are extracted separately: one row per original clip,
    # with no augmentation and no registered-member label.
    if a.unknown_audio:
        unknown_items = load_unknown_audio(a.unknown_audio)
        unknown_rows = []
        for item in unknown_items:
            row = {"source_file": item["source_file"]}
            row.update(extract_features(item["y"]))
            unknown_rows.append(row)
        if unknown_rows:
            unknown_df = pd.DataFrame(unknown_rows)
            Path(a.unknown_out).parent.mkdir(parents=True, exist_ok=True)
            unknown_df.to_csv(a.unknown_out, index=False)
            print(f"Saved {a.unknown_out}: {len(unknown_df)} independent outsider clips "
                  f"(one row per original recording; no augmentation).")
        else:
            print(f"[warn] no usable outsider recordings found in {a.unknown_audio}")


if __name__ == "__main__":
    main()
