"""Task 3 – audio pipeline.

Loads every team recording in ``audio/`` (named ``<member>_<phrase>.wav``),
cleans it, applies augmentations, extracts librosa features and writes
``data/audio_features.csv``.

The same ``load_clean`` + ``extract_features`` functions are used at
prediction time, so the CLI app can call ``verify_voice(path)`` on a new
recording and get exactly the features the voice model was trained on.

Run from the repo root:  python src/audio_pipeline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "audio"
UNAUTHORIZED_DIR = AUDIO_DIR / "unauthorized"
FEATURES_CSV = PROJECT_ROOT / "data" / "audio_features.csv"

SR = 16000            # every clip is resampled to this rate. 16 kHz (content up to 8 kHz)
                      # is the lowest rate among the phones used, so all clips cover the
                      # same frequency band and the model cannot learn the microphone.
TRIM_DB = 25          # silence below (peak - 25 dB) at the edges is trimmed
TARGET_PEAK = 0.9     # peak-normalisation level (fixes very quiet recordings)
N_MFCC = 13
SEED = 42
PHRASES = ("yes_approve", "confirm_transaction")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".mp4", ".mov", ".ogg", ".flac")


# ---------------------------------------------------------------- loading
def parse_name(path):
    """'robert_yes_approve.wav' -> ('robert', 'yes_approve')."""
    stem = Path(path).stem.lower()
    for phrase in PHRASES:
        if stem.endswith("_" + phrase):
            return stem[: -len(phrase) - 1], phrase
    raise ValueError(f"{Path(path).name}: expected <member>_<{'|'.join(PHRASES)}>")


def load_raw(path):
    """Load a recording as mono float audio at SR, without any cleaning."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def clean(y):
    """Trim leading/trailing silence, remove DC offset and peak-normalise."""
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    return y * (TARGET_PEAK / peak) if peak > 0 else y


def load_clean(path):
    return clean(load_raw(path))


def list_recordings(folder=AUDIO_DIR):
    """Every audio file directly inside `folder` (not sub-folders)."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in AUDIO_EXTS)


def quality_report(paths):
    """One row per recording: length, loudness, clipping, noise, pitch."""
    rows = []
    for path in paths:
        y = load_raw(path)
        trimmed, _ = librosa.effects.trim(y, top_db=TRIM_DB)
        rms = librosa.feature.rms(y=y)[0]
        f0, _, _ = librosa.pyin(trimmed, fmin=60, fmax=400, sr=SR)
        member, phrase = parse_name(path)
        peak = float(np.max(np.abs(y)))
        rows.append({
            "member": member, "phrase": phrase,
            "duration_s": round(len(y) / SR, 2),
            "speech_s": round(len(trimmed) / SR, 2),
            "peak": round(peak, 3),
            "clipped": peak >= 0.99,
            # loud frames (speech) vs quiet frames (background) in dB
            "snr_db": round(float(20 * np.log10(np.percentile(rms, 90) / (np.percentile(rms, 10) + 1e-10))), 1),
            "median_pitch_hz": round(float(np.nanmedian(f0)), 0) if np.any(~np.isnan(f0)) else np.nan,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------- augmentation
def add_noise(y, snr_db=20, rng=None):
    """Add white noise at a fixed signal-to-noise ratio."""
    rng = rng or np.random.default_rng(SEED)
    noise = rng.standard_normal(len(y))
    scale = np.sqrt(np.mean(y ** 2) / (10 ** (snr_db / 10)) / np.mean(noise ** 2))
    return y + scale * noise


def time_shift(y, seconds=0.2):
    """Move the speech later by padding silence at the start."""
    return np.concatenate([np.zeros(int(seconds * SR)), y])


AUGMENTATIONS = {
    "original":     lambda y: y,
    "pitch_up":     lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=2),
    "pitch_down":   lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-2),
    "speed_up":     lambda y: librosa.effects.time_stretch(y, rate=1.1),
    "slow_down":    lambda y: librosa.effects.time_stretch(y, rate=0.9),
    "noise":        add_noise,
    "volume_down":  lambda y: 0.5 * y,
    "time_shift":   time_shift,
}


# --------------------------------------------------------------- features
def extract_features(y):
    """MFCC 1-13 (mean + std), spectral roll-off, RMS energy and pitch (mean + std)."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)[0]
    rms = librosa.feature.rms(y=y)[0]
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR)   # fundamental frequency per frame
    f0 = f0[~np.isnan(f0)]                                 # keep voiced frames only
    feats = {f"mfcc_{i + 1}_mean": float(mfcc[i].mean()) for i in range(N_MFCC)}
    feats.update({f"mfcc_{i + 1}_std": float(mfcc[i].std()) for i in range(N_MFCC)})
    feats.update({
        "rolloff_mean": float(rolloff.mean()), "rolloff_std": float(rolloff.std()),
        "energy_mean": float(rms.mean()), "energy_std": float(rms.std()),
        "pitch_mean": float(f0.mean()) if f0.size else 0.0,
        "pitch_std": float(f0.std()) if f0.size else 0.0,
    })
    return feats


FEATURE_COLUMNS = list(extract_features(np.random.default_rng(0).standard_normal(SR)).keys())


def build_feature_table(paths):
    """One row per (recording, augmentation)."""
    rows = []
    for path in paths:
        member, phrase = parse_name(path)
        y = load_clean(path)
        for aug_name, augment in AUGMENTATIONS.items():
            rows.append({"member": member, "phrase": phrase, "augmentation": aug_name,
                         **extract_features(augment(y))})
    return pd.DataFrame(rows, columns=["member", "phrase", "augmentation"] + FEATURE_COLUMNS)


# ------------------------------------------------------------- prediction
def verify_voice(path, model, threshold, claimed_member=None):
    """Predict who is speaking in `path`.

    Returns a dict with the predicted member, its probability and whether
    access is granted. Access requires the top probability to reach
    `threshold` and, if `claimed_member` is given (e.g. from face
    recognition), the prediction to match it.
    """
    X = pd.DataFrame([extract_features(load_clean(path))])[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[0]
    best = int(np.argmax(proba))
    member, confidence = str(model.classes_[best]), float(proba[best])
    granted = confidence >= threshold and (claimed_member is None or member == claimed_member)
    return {"predicted_member": member, "confidence": confidence, "granted": bool(granted)}


if __name__ == "__main__":
    paths = list_recordings()
    print(quality_report(paths).to_string(index=False))
    table = build_feature_table(paths)
    FEATURES_CSV.parent.mkdir(exist_ok=True)
    table.to_csv(FEATURES_CSV, index=False)
    print(f"\nWrote {FEATURES_CSV.relative_to(PROJECT_ROOT)}: {table.shape[0]} rows x {table.shape[1]} columns, "
          f"{int(table.isna().sum().sum())} missing values")
