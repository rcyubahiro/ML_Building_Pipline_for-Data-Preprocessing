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
| 3 | Audio collection, augmentation, features (`audio_features.csv`), voice model | Aderline |
| 4 | Models: face, voice, product | _name_ |
| 6 | System demo (CLI app, unauthorized attempt, video) | _name_ |

Team members whose face and voice are in the dataset: **Elox, Aderline, Sandrine**.

## Repository layout

```
data/        source workbooks, clean_merged_customer_dataset.csv, image_features.csv, audio_features.csv, member_customer_map.csv
images/      <member>_<neutral|smiling|surprised>.jpg
images/unauthorized/   photo(s) of someone NOT on the team (demo only)
audio/       <member>_<yes_approve|confirm_transaction>.wav
audio/unauthorized/    recording(s) of someone NOT on the team (demo only)
notebooks/   one notebook per task (task1_customer_data_eda_cleaning_product_model.ipynb, task2_image_processing.ipynb, ...)
src/         pipelines and the CLI app for the other system stages
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
jupyter nbconvert --to notebook --execute notebooks/task1_customer_data_eda_cleaning_product_model.ipynb
python src/image_pipeline.py        # -> data/image_features.csv
python src/audio_pipeline.py        # -> data/audio_features.csv
python src/train_face.py
python src/train_voice.py
python src/train_product.py
python src/app.py --demo             # authorized + unauthorized attempts
```

The data-lead notebook writes `data/clean_merged_customer_dataset.csv`,
`reports/clean_merged_customer_summary_statistics.csv`,
`reports/clean_merged_customer_dataset_info.txt`,
`reports/clean_merged_customer_variable_types.csv`, three named plots, and
`reports/product_recommendation_model_metrics.json`. The merge uses an inner
join after normalizing `customer_id_new` and `customer_id_legacy`. Product
metrics include accuracy, macro-F1, and log loss. Duplicate social profiles
for one normalized customer are reduced to one profile before joining to avoid
many-to-many row inflation.

Integration evidence is documented in `reports/member4_integration_report.md`.
The shared model results table is `reports/model_evaluation.csv`, and the multimodal
flowchart is `reports/multimodal_flowchart.mmd`. The application validates all three
model bundles at startup and only runs the product model after face and voice checks
pass. `data/member_customer_map.csv` links each demo member to a customer record.

Notebooks can be opened from anywhere; they switch to the repo root automatically.

## Working together (Git)

1. `git clone` the repo, then create your own branch: `git checkout -b task-3-audio`.
2. Only edit files that belong to your task. If you need to change someone else's file, tell them.
3. Commit small and often: `git add <your files>` then `git commit -m "Task 3: add audio features"`.
4. `git pull --rebase origin main` before you push, then `git push origin task-3-audio`, then open a pull request into `main`.
5. Shared files (`README.md`, `requirements.txt`, `src/common.py`): pull first and keep edits small.
6. Do not commit `.venv/`, `__pycache__/`, or large model files.

## Privacy

This repo contains photos and voice recordings of team members. We Kept it **private** And It will be only shared among team members and instructor(s).

## Contributions

Each member adds a short paragraph on their own contribution below.

- **Task 2 (Robert):** collected and standardized the facial images (3 expressions per member), preprocessed them (face crop, 128x128, grayscale), applied five augmentations (rotation +/-15 degrees, flip, Gaussian blur, brightness), extracted intensity-histogram and HOG features into `data/image_features.csv`, and wrote `src/image_pipeline.py` and `notebooks/task2_image_processing.ipynb`.
- **Task 1 (Data Lead):** completed in `notebooks/task1_customer_data_eda_cleaning_product_model.ipynb`; cleaned and merged the supplied workbooks, generated EDA summaries and labeled plots under `reports/`, and trained the Random Forest product model.
- **Task 3 (Aderline):** converted the team's phone recordings (AAC/Opus, 16–48 kHz) to mono `.wav` with the repo naming, checked quality (clipping, SNR, length), trimmed silence, peak-normalised and resampled everything to 16 kHz; plotted a waveform and spectrogram per recording; applied seven augmentations (pitch ±2 semitones, speed ±10%, noise, volume, time shift); extracted MFCC 1–13 (mean/std), roll-off, RMS energy and pitch into `data/audio_features.csv`; trained the Random Forest voice verification model with a leave-one-phrase-out evaluation (`src/audio_pipeline.py`, `src/train_voice.py`, `notebooks/task3_audio_processing.ipynb`).
