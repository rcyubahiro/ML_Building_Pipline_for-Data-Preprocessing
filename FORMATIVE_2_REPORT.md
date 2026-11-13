# Formative 2 - Data Preprocessing
## Multimodal Authentication and Product Recommendation System

**Team:** Team 27  
**Project deadline:** 18 November 2026, 11:59 pm  

## 1. Introduction

This project combines three machine-learning tasks into one system. First, the system identifies a user from a facial image. Second, it verifies the user's voice. Finally, after both checks pass, it recommends a product using customer transaction and social-media data.

The purpose of the system is to demonstrate how different types of data can be combined in one authentication and recommendation workflow. A genuine user should receive a product recommendation, while an unauthorised user should be denied access.

The main stages are:

```text
Face recognition -> Voice verification -> Product recommendation
       fail              fail
   Access denied      Access denied
```

## 2. Data sources and project files

The project uses:

- Customer transaction data from `data/customer_transactions.xlsx`
- Customer social-profile data from `data/customer_social_profiles.xlsx`
- Facial images stored in `images/`
- Voice recordings stored in `audio/`
- Separate notebooks for data, image and audio processing
- Python pipelines in `src/`
- Generated feature files in `data/`
- Models and plots in `models/` and `reports/`

The main notebooks are:

- `notebooks/task1_customer_data_eda_cleaning_product_model.ipynb`
- `notebooks/task2_image_processing.ipynb`
- `notebooks/task3_audio_processing.ipynb`

## 3. Exploratory data analysis

The data notebook begins by loading the social-profile and transaction workbooks. It displays their shapes, summary statistics and variable types. The analysis checks which variables are numeric, categorical or date-related.

Three important plots were created:

1. **Purchase amount distribution:** This shows how purchase values are spread and whether the data is skewed.
2. **Purchase amount outlier plot:** This shows the median, spread and unusually high or low purchases.
3. **Correlation heatmap:** This shows relationships between numeric variables such as purchase amount, rating, engagement and purchase interest.

The plots are saved in `reports/`. Each plot has a title, axis labels and an explanation in the notebook. These results help explain the data before modelling rather than treating the input tables as ready for training.

## 4. Data cleaning and merge validation

The customer data was prepared through the following steps:

1. Duplicate rows were identified and removed.
2. Numeric missing values were handled using the median.
3. Categorical missing values were handled using the most common value.
4. Date and numeric columns were converted to suitable types.
5. Customer identifiers were normalised so that IDs such as `A151` could match legacy numeric IDs such as `151`.
6. Duplicate social profiles were reduced before joining.
7. The cleaned transaction and social data were merged using the normalised customer ID.
8. The result was checked for row-count changes, missing values and duplicate records.

The cleaned dataset is saved as `data/clean_merged_customer_dataset.csv`. It currently contains 117 rows and 11 columns. The join logic is documented because an incorrect many-to-many join could create duplicate transactions and misleading model results.

## 5. Image data preparation

The image pipeline checks filenames, reads the images, converts them to a consistent format and prepares them for feature extraction. The required expressions are:

- Neutral
- Smiling
- Surprised

For the current image dataset, Aderline, Elox and Sandrine each have the three required expressions. The files follow the naming format:

```text
<member>_neutral.jpg
<member>_smiling.jpg
<member>_surprised.jpg
```

The image pipeline applies several transformations, including rotation, horizontal flipping, blurring and brightness adjustment. It then extracts grayscale histograms, grayscale statistics and HOG features.

The generated file `data/image_features.csv` currently contains 54 rows and 346 columns, with no missing values. The six variants per source image provide more training examples than the original images alone.

**Important limitation:** the current image dataset contains three members. If the assignment requires all four team members to be represented, the fourth member's three facial images must be added before the final submission.

## 6. Audio data preparation

Each recorded team member has two phrases:

- `yes_approve`
- `confirm_transaction`

The audio pipeline checks the recordings for length, clipping, noise and pitch. It trims unnecessary silence, converts the recordings to a common sample rate and prepares them for feature extraction.

The notebook includes waveform and spectrogram plots. The waveform shows the amplitude over time, while the spectrogram shows how the frequency content changes during speech. These plots help explain whether the recordings contain silence, noise or clipping.

The audio pipeline applies pitch changes, speed changes, noise, volume changes and time shifts. It extracts:

- MFCC means and standard deviations
- Spectral roll-off
- Energy
- Pitch

The generated `data/audio_features.csv` contains 64 rows and 35 columns. It has no missing values and no duplicate rows.

## 7. Model implementation

### Face-recognition model

The face model uses the extracted image features to predict the member in the image. It is evaluated using leave-one-expression-out validation so that one expression is held out during testing.

### Voice-verification model

The voice model uses a scaled Random Forest classifier with audio features. It uses leave-one-phrase-out validation. This is more meaningful than randomly splitting augmented rows because augmented versions of the same recording are very similar.

### Product-recommendation model

The product model uses cleaned customer and social-profile features to predict the product category. Logistic Regression and Random Forest models are compared, and the better model is selected.

The models are saved with consistent names:

- `models/face_model.pkl`
- `models/voice_model.pkl`
- `models/product_model.pkl`

## 8. Evaluation results

The current results are summarised below. The full table is saved in `reports/model_evaluation.csv`.

| Model | Accuracy | Macro-F1 | Log loss | Validation |
|---|---:|---:|---:|---|
| Face recognition | 1.000 | 1.000 | 0.013 | Leave-one-expression-out |
| Voice verification | 0.719 | 0.651 | 0.931 | Leave-one-phrase-out |
| Product recommendation | 0.292 | 0.246 | 1.960 | Stratified 20% hold-out |

The face model performs strongly on the current three-member image dataset. The voice model gives a reasonable but weaker result, which is expected because voice recordings vary by phrase and recording conditions.

The product model has the lowest score. This means that its output should be described as a prediction rather than a guaranteed recommendation. The final discussion should mention the small dataset, class overlap and limited number of transactions as possible reasons for the weaker performance.

The voice confusion matrices are saved in `reports/figures/task3_voice_confusion_matrices.png`.

## 9. Multimodal authentication logic

The system applies the checks in a fixed order:

1. The face image is analysed.
2. If the face is not recognised, the system prints an access-denied message and stops.
3. If the face is recognised, the voice recording is analysed.
4. The predicted speaker must match the person identified by the face model.
5. If the voice does not match or confidence is too low, access is denied.
6. Only after both checks pass does the system load the customer's record.
7. The product model predicts and displays the recommended category.

This order is important because the product recommendation must not be shown to an unauthorised user. The flowchart is available in `reports/multimodal_flowchart.mmd`.

## 10. System simulation

The complete simulation is run with:

```bash
python src/app.py --demo
```

The demo was repeated three times. The final behaviour was consistent:

```text
Authorised face and authorised voice:
Access granted and product recommendation displayed.

Unauthorised face:
Access denied before the voice and product stages.

Authorised face and unauthorised voice:
Access denied before the product stage.

Demo finished: 1 approved, 2 denied.
```

For the authorised example, the system identified Aderline, verified the matching voice and predicted **Books** as the top product category.

## 11. Discussion and limitations

The system demonstrates the required end-to-end idea, but the results should be interpreted carefully.

First, the image dataset currently covers three members rather than the complete four-person team. The missing member's images should be added and the face model retrained.

Second, the product model's accuracy and macro-F1 are low. More customer records, more balanced product classes and stronger feature engineering could improve the recommendation stage.

Third, the voice threshold was increased to `0.75` for the transaction gate so that the current outsider recordings are rejected. A stricter threshold can also reject genuine recordings with low confidence, so all authorised members should be tested before submission.

Finally, the OpenCV installation used during testing did not provide the Haar cascade detector API. The image pipeline therefore used its documented full-image fallback. A compatible OpenCV installation should be used for the final run if face-cropping evidence is required.

## 12. Reproducibility

From the project root:

```bash
python -m pip install -r requirements.txt
python src/image_pipeline.py
python src/train_face.py
python src/audio_pipeline.py
python src/train_voice.py
python src/train_product.py
python src/app.py --demo
```

Before exporting the final PDF, the team should run the notebooks from a fresh kernel and save the outputs.

## 13. Rubric compliance checklist

- [x] Summary statistics, variable types and at least three labelled EDA plots
- [x] Null, duplicate, type and merge handling documented
- [ ] Three expressions confirmed for every team member
- [x] Image augmentation and image feature extraction
- [x] Two audio phrases per current member
- [x] Audio cleaning, visualisation, augmentation and feature extraction
- [x] Face, voice and product model implementations
- [x] Accuracy, F1-score and loss table
- [x] Multimodal authentication logic and flowchart
- [x] Authorised and unauthorised simulation
- [x] Code, notebooks, CSV files and model files organised
- [ ] Final clean run after adding any missing team data

## 14. Conclusion

This project shows how image, audio and tabular data can be processed and combined in one practical authentication system. The preprocessing pipelines create reusable features, the three models are trained and evaluated, and the final application demonstrates both successful authentication and access denial.

The most important remaining step is to complete the image data for all team members and rerun the final models and notebooks. After that, the team should export this report together with the final notebook, source code, CSV files, plots and model files.
