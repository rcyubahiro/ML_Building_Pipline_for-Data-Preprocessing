# Formative 2: Multimodal Data Preprocessing

A user identity and product recommendation system. A user is authenticated by **face**, then **voice**, before a personalised product prediction is shown.

```
Start -> Face recognition -> Product model -> Voice validation -> Display predicted product
              | fail                                | fail
        Access Denied                          Access Denied
```

## Team and ownership

| Task | What | Owner |
|---|---|---|
| 1 | Data merge + EDA (`customer_social_profiles` + `customer_transactions`) | _name_ |
| 2 | Image collection, augmentation, features (`image_features.csv`) | Chriss |
| 3 | Audio collection, augmentation, features (`audio_features.csv`) | _name_ |
| 4 | Models: face, voice, product | _name_ |
| 6 | System demo (CLI app, unauthorized attempt, video) | _name_ |

Team members whose face and voice are in the dataset: **Elox, Aderline, Sandrine**.

## Repository layout

```
data/        raw CSVs, merged_dataset.csv, image_features.csv, audio_features.csv, member_customer_map.csv
images/      <member>_<neutral|smiling|surprised>.jpg
images/unauthorized/   photo(s) of someone NOT on the team (demo only)
audio/       <member>_<yes_approve|confirm_transaction>.wav
audio/unauthorized/    recording(s) of someone NOT on the team (demo only)
notebooks/   one notebook per task (task2_image_processing.ipynb, ...)
src/         pipelines, training scripts, and the CLI app
models/      trained models (*.pkl are git-ignored; rebuild with the train scripts)
reports/     figures/ and metrics.json
```

### Naming rules (the code depends on these)
- Member names are lowercase: `elox`, `aderline`, `sandrine`.
- Images: `elox_neutral.jpg`, `elox_smiling.jpg`, `elox_surprised.jpg`
- Audio: `elox_yes_approve.wav`, `elox_confirm_transaction.wav`
- `data/member_customer_map.csv` links each member to a customer ID in the merged dataset. Fill in the blanks.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

```bash
python src/merge_data.py --social data/customer_social_profiles.csv --transactions data/customer_transactions.csv
python src/image_pipeline.py        # -> data/image_features.csv
python src/audio_pipeline.py        # -> data/audio_features.csv
python src/train_face.py
python src/train_voice.py
python src/train_product.py
python src/app.py --demo            # authorized + unauthorized attempts
```

Notebooks can be opened from anywhere; they switch to the repo root automatically.

## Working together (Git)

1. `git clone` the repo, then create your own branch: `git checkout -b task-3-audio`.
2. Only edit files that belong to your task. If you need to change someone else's file, tell them.
3. Commit small and often: `git add <your files>` then `git commit -m "Task 3: add audio features"`.
4. `git pull --rebase origin main` before you push, then `git push origin task-3-audio`, then open a pull request into `main`.
5. Shared files (`README.md`, `requirements.txt`, `src/common.py`): pull first and keep edits small.
6. Do not commit `.venv/`, `__pycache__/`, or large model files.

## Privacy

This repo contains photos and voice recordings of team members. Keep the repository **private** and share it only with the team and instructor.

## Contributions

Each member adds a short paragraph on their own contribution below.

- **Task 2 (Chriss):** collected and standardized the facial images (3 expressions per member), preprocessed them (face crop, 128x128, grayscale), applied five augmentations (rotation +/-15 degrees, flip, Gaussian blur, brightness), extracted intensity-histogram and HOG features into `data/image_features.csv`, and wrote `src/image_pipeline.py` and `notebooks/task2_image_processing.ipynb`.
