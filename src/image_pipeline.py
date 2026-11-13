"""Task 2: load face images, display samples, augment, extract features -> image_features.csv.

Expected naming (consistent, as the rubric asks):
    images/<member>_<expression>.<jpg|jpeg|png>
    expression in {neutral, smiling, surprised}
e.g. images/chriss_neutral.jpg

Run from the repo root:
    python src/image_pipeline.py --images images --out data/image_features.csv
"""
import argparse
import re
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog

EXPRESSIONS = ("neutral", "smiling", "surprised")
EXTS = {".jpg", ".jpeg", ".png"}
SIZE = 128  # faces are resized to SIZE x SIZE before feature extraction

def _load_cascade():
    name = "haarcascade_frontalface_default.xml"
    candidates = []
    if getattr(cv2, "data", None):
        candidates.append(Path(cv2.data.haarcascades) / name)
    candidates.append(Path(__file__).resolve().parent.parent / "models" / name)  # repo fallback
    for c in candidates:
        if c.exists():
            cc = cv2.CascadeClassifier(str(c))
            if not cc.empty():
                return cc
    return cv2.CascadeClassifier()


_cascade = _load_cascade()
if _cascade.empty():
    print("[warn] Haar face cascade not found; using the full image instead of a face crop.")


def parse_name(path):
    m = re.match(r"^(?P<member>.+)_(?P<expr>neutral|smiling|surprised)$", path.stem, re.I)
    if not m:
        return None
    return m["member"].lower(), m["expr"].lower()


def load_images(folder):
    items, skipped = [], []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        parsed = parse_name(p)
        img = cv2.imread(str(p))
        if parsed is None or img is None:
            skipped.append(p.name)
            continue
        items.append({"member": parsed[0], "expression": parsed[1], "path": p, "bgr": img})
    if skipped:
        print(f"[warn] skipped (bad name or unreadable): {skipped}")
    return items


def crop_face(bgr):
    """Crop the largest detected face; fall back to the full image if none is found."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = [] if _cascade.empty() else _cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        bgr = bgr[y:y + h, x:x + w]
    # every face is converted to grayscale so colour vs black-and-white photos cannot leak identity
    return to_gray(cv2.resize(bgr, (SIZE, SIZE)))


# ---------- augmentations ----------
def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def to_gray(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


AUGMENTATIONS = {
    "original": lambda im: im,
    "rotate_pos15": lambda im: rotate(im, 15),
    "rotate_neg15": lambda im: rotate(im, -15),
    "flip_horizontal": lambda im: cv2.flip(im, 1),
    "gaussian_blur": lambda im: cv2.GaussianBlur(im, (7, 7), 0),
    "brightness_up": lambda im: cv2.convertScaleAbs(im, alpha=1.0, beta=40),
}


# ---------- features ----------
def extract_features(bgr):
    feats = {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 1) grayscale intensity histogram: 16 bins, normalised (all faces are grayscale-standardised)
    h = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    h = h / (h.sum() + 1e-9)
    for j, v in enumerate(h):
        feats[f"hist_gray_{j}"] = float(v)
    feats["gray_mean"] = float(gray.mean())
    feats["gray_std"] = float(gray.std())
    # 2) HOG descriptor of the 64x64 grayscale face = compact shape "embedding"
    small = cv2.resize(gray, (64, 64))
    vec = hog(small, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    for j, v in enumerate(vec):
        feats[f"hog_{j}"] = float(v)
    return feats


def show_samples(items, out_png):
    members = sorted({i["member"] for i in items})
    fig, axes = plt.subplots(len(members), len(EXPRESSIONS),
                             figsize=(3 * len(EXPRESSIONS), 3 * len(members)), squeeze=False)
    for r, m in enumerate(members):
        for c, e in enumerate(EXPRESSIONS):
            ax = axes[r][c]
            ax.axis("off")
            match = [i for i in items if i["member"] == m and i["expression"] == e]
            if match:
                ax.imshow(cv2.cvtColor(crop_face(match[0]["bgr"]), cv2.COLOR_BGR2RGB))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_title(f"{m} - {e}", fontsize=9)
    fig.suptitle("Sample faces per member and expression")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def show_augmentations(item, out_png):
    face = crop_face(item["bgr"])
    fig, axes = plt.subplots(1, len(AUGMENTATIONS), figsize=(3 * len(AUGMENTATIONS), 3.2))
    for ax, (name, fn) in zip(axes, AUGMENTATIONS.items()):
        ax.imshow(cv2.cvtColor(fn(face), cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Augmentations: {item['member']} - {item['expression']}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    matplotlib.use("Agg")  # script mode: write PNGs, no window (notebooks keep their own backend)
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="data/image_features.csv")
    ap.add_argument("--plots", default="reports/figures")
    a = ap.parse_args()

    items = load_images(a.images)
    if not items:
        raise SystemExit(f"No valid images found in {a.images}. Expected <member>_<expression>.jpg")

    members = sorted({i["member"] for i in items})
    print(f"Loaded {len(items)} images for {len(members)} members: {members}")
    for m in members:
        have = {i["expression"] for i in items if i["member"] == m}
        miss = set(EXPRESSIONS) - have
        if miss:
            print(f"[warn] {m} is missing expressions: {sorted(miss)}")

    Path(a.plots).mkdir(parents=True, exist_ok=True)
    show_samples(items, Path(a.plots) / "image_samples.png")
    show_augmentations(items[0], Path(a.plots) / "image_augmentations_example.png")

    rows = []
    for it in items:
        face = crop_face(it["bgr"])
        for aug_name, fn in AUGMENTATIONS.items():
            row = {"member": it["member"], "expression": it["expression"],
                   "source_file": it["path"].name, "augmentation": aug_name}
            row.update(extract_features(fn(face)))
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {a.out}: {df.shape[0]} rows x {df.shape[1]} cols "
          f"({len(AUGMENTATIONS)} variants per source image)")


if __name__ == "__main__":
    main()
