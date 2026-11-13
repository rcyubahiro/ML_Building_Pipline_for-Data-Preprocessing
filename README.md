# Formative 2: Multimodal Data Preprocessing

## 1. Project Overview

This project implements a multimodal user authentication and product recommendation system using machine learning.

The application combines three components:

1. **Face recognition:** Identifies an enrolled team member using facial image features.
2. **Voice recognition:** Evaluates the user's voice against enrolled speaker profiles.
3. **Product recommendation:** Predicts product categories using customer transaction data.

The system uses sequential authentication. A user must pass face recognition and voice verification before the product recommendation component can be accessed.

### System Architecture

```text
       Start Transaction
              |
              v
       Capture Face Image
              |
              v
       Face Recognition
              |
         Face Accepted?
          /       \
        No         Yes
        |           |
        v           v
   Access Denied  Capture Voice
                    |
                    v
              Voice Verification
                    |
               Voice Accepted?
                /       \
              No         Yes
              |           |
              v           v
         Access Denied  Product Model
                            |
                            v
                    Generate Recommendations
                            |
                            v
                    Display Recommendations
```

**Security rule:** Product recommendations must not be generated or displayed unless both authentication stages pass.

The application follows a fail-closed approach. If voice outsider rejection is not sufficiently calibrated, access is denied rather than approving a user based only on classification confidence.

---

## 2. Team and Task Ownership

| Task | Responsibility | Contributor |
|---|---|---|
| Task 1 | Customer data cleaning, merging, EDA and product model | Sandrine / Data Lead |
| Task 2 | Facial image collection, preprocessing and feature extraction | Robert |
| Task 3 | Audio collection, preprocessing and voice feature extraction | Aderline |
| Task 4 | Model integration, authentication workflow and validation | Eloi (Elox) |
| Task 6 | System demonstration and integration presentation | Eloi (Elox) |

The enrolled members represented in the facial and voice datasets are:

- Elox
- Aderline
- Robert
- Sandrine

---

## 3. Repository Structure

```text
ML_Building_Pipline_for-Data-Preprocessing/
│
├── audio/
│   ├── unauthorized/
│   └── enrolled member recordings
│
├── images/
│   ├── unauthorized/
│   └── enrolled member images
│
├── data/
│   ├── customer_social_profiles.xlsx
│   ├── customer_transactions.xlsx
│   ├── merged_dataset.csv
│   ├── image_features.csv
│   ├── audio_features.csv
│   ├── unknown_audio_features.csv
│   └── member_customer_map.csv
│
├── models/
│   ├── face_model.pkl
│   ├── voice_model.pkl
│   └── product_model.pkl
│
├── notebooks/
│   ├── task1_customer_data_eda_cleaning_product_model.ipynb
│   ├── task2_image_processing.ipynb
│   └── task3_audio_processing.ipynb
│
├── reports/
│   ├── metrics.json
│   ├── voice_outsider_evaluation.csv
│   ├── task3_face_confusion_matrix.png
│   ├── task3_voice_confusion_matrices.png
│   ├── task3_product_confusion_matrix.png
│   └── figures/
│
├── src/
│   ├── app.py
│   ├── common.py
│   ├── image_pipeline.py
│   ├── audio_pipeline.py
│   ├── merge_data.py
│   ├── train_face.py
│   ├── train_voice.py
│   ├── train_product.py
│   ├── generate_face_product_confusion_matrices.py
│   └── generate_voice_confusion_matrix.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Data Preparation

### 4.1 Customer Data

The customer data preparation process uses two supplied Excel workbooks:

- `customer_social_profiles.xlsx`
- `customer_transactions.xlsx`

The merge pipeline performs data cleaning, duplicate handling, customer ID matching, and dataset integration.

The resulting dataset contains:

| Description | Result |
|---|---:|
| Transaction records | 150 |
| Matched social profiles | 117 |
| Unmatched transactions | 33 |
| Missing values in merged dataset | 0 |
| Final dataset columns | 20 |

The merged dataset is saved as:

`data/merged_dataset.csv`

All 150 transaction records are preserved.

### 4.2 Facial Image Processing

The image pipeline processes facial images belonging to the four enrolled members.

Processing includes face detection, image standardization, augmentation, and feature extraction.

- Original images: 12
- Enrolled members: 4
- Generated feature rows: 72
- Features per row: 346 columns, including metadata

The extracted features are saved in:

`data/image_features.csv`

### 4.3 Audio Processing

The audio pipeline processes the enrolled members' voice recordings and generates augmented audio samples.

Feature extraction includes MFCC statistics and additional audio characteristics such as RMS energy, spectral roll-off, and pitch.

- Original enrolled recordings: 8
- Enrolled members: 4
- Generated feature rows: 64
- Features per row: 36 columns, including metadata

The extracted features are saved in:

`data/audio_features.csv`

Unauthorized recordings are processed separately and saved in:

`data/unknown_audio_features.csv`

Augmented recordings are not treated as independent original recordings during grouped validation.

---

## 5. Model Development and Evaluation

Three machine-learning components were trained and evaluated.

Random Forest was selected for each component after comparison with Logistic Regression.

The Random Forest configuration explicitly uses:

- `n_estimators = 300`
- `random_state = 42`

### 5.1 Face Recognition

The face recognition model identifies enrolled members from facial image features.

Validation uses grouped cross-validation based on facial expression, keeping each expression group together.

| Metric | Random Forest |
|---|---:|
| Accuracy | 95.8% |
| Macro F1-score | 95.8% |
| Log loss | 0.632 |

Selected model:

`models/face_model.pkl`

### 5.2 Voice Recognition

The voice model identifies enrolled speakers using extracted audio features.

Validation groups augmented samples by their original source recording to reduce the risk of training and validation leakage.

| Metric | Random Forest |
|---|---:|
| Accuracy | 46.9% |
| Macro F1-score | 47.6% |
| Log loss | 18.306 |

Selected model:

`models/voice_model.pkl`

**Important limitation:** There are only two independent original recordings per enrolled member. The available outsider recordings are insufficient for reliable open-set rejection calibration.

The system therefore sets `unknown_calibrated` to false and denies access when voice authentication cannot be safely established.

### 5.3 Product Classification

The product model predicts product categories using the merged customer transaction dataset.

The evaluation uses an 80/20 train-test split, with a 30-record holdout test set.

| Metric | Random Forest |
|---|---:|
| Holdout accuracy | 16.7% |
| Macro F1-score | 14.9% |
| Log loss | 1.907 |
| Majority-class baseline accuracy | 23.3% |

Selected model:

`models/product_model.pkl`

The holdout accuracy is below the majority-class baseline. This indicates that the product model requires further improvement before it can be considered reliable.

### 5.4 Consolidated Evaluation

| Component | Selected algorithm | Accuracy | Macro F1 |
|---|---|---:|---:|
| Face | Random Forest | 95.8% | 95.8% |
| Voice | Random Forest | 46.9% | 47.6% |
| Product | Random Forest | 16.7% | 14.9% |

The detailed results are stored in:

`reports/metrics.json`

Confusion matrices are available in the `reports/` directory.

---

## 6. System Integration

The integration stage combines the three trained models into a single command-line application.

The application executes the following process:

1. Receives a facial image.
2. Uses the face model to identify the closest enrolled member.
3. Rejects the transaction if face recognition fails.
4. Receives a voice recording if the face is accepted.
5. Evaluates the voice recording using the voice model.
6. Rejects the transaction if voice verification fails or outsider rejection is not calibrated.
7. Runs product recommendations only after both authentication stages pass.

The application does not generate recommendations before authentication succeeds.

### Member-to-Customer Mapping

The file `data/member_customer_map.csv` is reserved for linking enrolled members to customer records.

The customer IDs are currently unverified and must remain blank until the correct relationships are confirmed.

The application must not invent member-to-customer relationships or use unverified customer histories for personalized recommendations.

If authentication succeeds while no verified mapping is available, the application can use its general popularity-based recommendation fallback.

---

## 7. Integration Testing

The following scenarios were executed through the command-line application:

| Test | Face result | Voice result | Final outcome |
|---|---|---|---|
| Enrolled member with genuine voice | Accepted | Denied because outsider rejection is uncalibrated | Access denied |
| Unauthorized facial image | Rejected | Not evaluated | Access denied |
| Enrolled member's face with outsider voice | Accepted | Denied | Access denied |

All three test transactions were denied.

These tests demonstrate that:

- Face rejection prevents further authentication.
- Voice verification is performed only after face acceptance.
- An accepted face alone is insufficient for access.
- An outsider voice does not receive access.
- Recommendations are not displayed after failed authentication.

A successful end-to-end recommendation transaction has not yet been demonstrated because the voice rejection calibration is incomplete.

---

## 8. Limitations and Future Improvements

The current system has several limitations:

1. **Limited voice recordings:** More independent recordings are needed for each enrolled speaker.
2. **Outsider calibration:** Additional independent outsider recordings are required before reliable unknown-speaker rejection can be calibrated.
3. **Voice recognition performance:** The current voice model has relatively low accuracy and macro F1.
4. **Product classification:** The product model performs below the majority-class baseline and requires further feature engineering, data analysis, and model evaluation.
5. **Customer mapping:** Verified member-to-customer IDs are not yet available, preventing reliable personalized recommendations.
6. **End-to-end validation:** A successful recommendation demonstration requires improved voice verification and a valid authentication path.

Future work should focus on collecting more independent audio data, improving the voice and product models, and verifying customer mappings.

---

## 9. Setup and Installation

The project uses Python 3.13.

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

---

## 10. Running the Project

Run the following commands from the project root.

### Step 1: Merge customer data

```powershell
python src/merge_data.py
```

### Step 2: Generate facial features

```powershell
python src/image_pipeline.py --images images --out data/image_features.csv
```

### Step 3: Generate audio features

```powershell
python src/audio_pipeline.py --audio audio --out data/audio_features.csv
```

To process unauthorized recordings separately:

```powershell
python src/audio_pipeline.py --audio audio --out data/audio_features.csv --unknown-audio audio/unauthorized --unknown-out data/unknown_audio_features.csv
```

### Step 4: Train the models

```powershell
python src/train_face.py
```

```powershell
python src/train_voice.py --features data/audio_features.csv --unknown-features data/unknown_audio_features.csv
```

```powershell
python src/train_product.py
```

### Step 5: Generate evaluation confusion matrices

```powershell
python src/generate_voice_confusion_matrix.py
```

```powershell
python src/generate_face_product_confusion_matrices.py
```

### Step 6: Run the integrated application

```powershell
python src/app.py --demo
```

The demo executes the available authorized and unauthorized test scenarios.

**Note:** With the current voice calibration limitation, even a genuine enrolled user may be denied. This is intentional fail-closed behavior, not evidence of successful authentication.

---

## 11. Individual Contribution — Integration Lead

**Eloi (Elox) — Integration Lead**

I integrated the face recognition, voice recognition, and product classification models into a single application workflow. I implemented sequential authentication so that face recognition is performed first, followed by voice verification, and product recommendations are only accessed after both authentication stages pass.

I also coordinated model loading and validation, reviewed model evaluation metrics, generated confusion matrices, implemented fail-closed voice authentication behavior, evaluated unauthorized access scenarios, and documented the limitations of the available training data. In addition, I maintained the integration documentation and verified that the trained models and evaluation outputs were available in the project repository.

---

## 12. Privacy and Data Handling

The repository contains facial images and voice recordings belonging to team members.

These materials are private project data and should only be shared with authorized team members and instructors.

Unauthorized images and recordings are included strictly for testing access rejection.

Large generated model files and local virtual environments should not be committed unless the assignment explicitly requires them.

---

## 13. Final Project Deliverables

The project includes:

- Customer data cleaning and merging pipeline.
- Customer exploratory data analysis notebook.
- Facial image preprocessing and feature extraction pipeline.
- Audio preprocessing and feature extraction pipeline.
- Face recognition model.
- Voice recognition model.
- Product classification model.
- Integrated command-line application.
- Model evaluation metrics.
- Face, voice, and product confusion matrices.
- Unauthorized access evaluation.
- Project documentation and individual contributions.

**Project status:** The three models and integration workflow are implemented. Authentication sequencing and denial scenarios have been tested. Further voice calibration and product model improvement are required before reliable end-to-end recommendations can be demonstrated.
