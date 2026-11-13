# Member 4 Report: Integration, Evaluation and System Simulation

## 1. My integration responsibility

As the Integration Lead, I brought together the face-recognition, voice-verification and product-recommendation parts of the project. My main responsibility was to make sure that the models could work together in one transaction flow instead of running as separate demonstrations.

I also checked the model files, prepared a common evaluation table, documented the authentication logic and tested both a genuine transaction and unauthorised attempts.

## 2. Model hand-off and checking

The project now uses these model files:

| Model | File | Algorithm and features | Validation method |
|---|---|---|---|
| Face recognition | `models/face_model.pkl` | Logistic Regression selected from Logistic Regression and Random Forest; histogram, grayscale and HOG image features | Leave-one-expression-out validation |
| Voice verification | `models/voice_model.pkl` | Random Forest with feature scaling; MFCC, pitch, roll-off and energy features | Leave-one-phrase-out validation |
| Product recommendation | `models/product_model.pkl` | Random Forest selected from Logistic Regression and Random Forest; cleaned customer and social-data features | Stratified 20% hold-out test |

The application checks the model files when it starts. It confirms that each file contains a trained model, its feature names and its class labels. It also checks that the model can produce probabilities before a transaction is attempted.

The demo customer links are stored in `data/member_customer_map.csv`. These links are currently used for integration testing and should be confirmed against the team's final customer assignments before submission.

## 3. Evaluation results

The complete results are saved in [`model_evaluation.csv`](model_evaluation.csv).

| Model | Accuracy | Macro-F1 | Log loss |
|---|---:|---:|---:|
| Face recognition | 1.000 | 1.000 | 0.013 |
| Voice verification | 0.719 | 0.651 | 0.931 |
| Product recommendation | 0.292 | 0.246 | 1.960 |

The face model performed strongly on the current three-member image dataset. This result must be updated if another member's facial images are added.

The voice model correctly recognised many of the team recordings, but its results were weaker than the face model. The voice notebook includes confusion matrices in `reports/figures/task3_voice_confusion_matrices.png`.

The product model has the lowest score. This means that the recommendation should be presented as a model prediction, not as a guaranteed correct product choice. The report should include this limitation honestly.

## 4. Multimodal authentication logic

The system uses two authentication checks before showing a recommendation:

1. The face image is analysed.
2. If the face is not recognised, access is denied immediately.
3. If the face is accepted, the voice recording is analysed.
4. The predicted speaker must match the person identified by the face model.
5. If the voice does not match or its confidence is too low, access is denied.
6. Only when both checks pass does the system load the customer's record and run the product model.
7. The predicted product is then displayed.

This prevents the product recommendation from being shown when authentication fails. The flowchart is available in [`multimodal_flowchart.mmd`](multimodal_flowchart.mmd).

## 5. System simulation

The complete demonstration can be run from the project root with:

```bash
python src/app.py --demo
```

The demonstration was run three times. The result was consistent:

```text
Authorised face and authorised voice:
Access granted and product recommendation displayed.

Unauthorised face:
Access denied before the voice and product stages.

Authorised face and unauthorised voice:
Access denied before the product stage.

Demo finished: 1 approved, 2 denied.
```

For the authorised example, the system identified Aderline, verified the matching voice and displayed **Books** as the top product recommendation.

For the unauthorised face example, the system rejected the face. For the unauthorised voice example, the face was accepted but the voice confidence was below the transaction threshold, so the system denied access.

The voice threshold used for the transaction gate is `0.75`. This threshold was chosen to reject the current outsider recordings. A stricter threshold can also reject genuine recordings with low confidence, so the team should test every authorised member before final submission.

## 6. Reproducibility instructions

Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python src/image_pipeline.py
python src/train_face.py
python src/audio_pipeline.py
python src/train_voice.py
python src/train_product.py
python src/app.py --demo
```

The application uses the cleaned customer file and the member-to-customer mapping file. The final team should confirm that the mapping reflects the intended customer accounts.

## 7. Current limitation and final check

The current image dataset contains three members: Aderline, Elox and Sandrine. If the assignment requires all four team members to have face images, the fourth member's three expressions must be added and the image features and face model must be regenerated.

The OpenCV installation used during testing did not provide the Haar cascade detector API. The image pipeline therefore used its documented full-image fallback and printed a warning. A compatible OpenCV installation should be used for the final run if face-cropping evidence is required.

Before submission, the team should:

- Confirm all required team images and audio recordings are present.
- Confirm the customer mapping.
- Rerun all three training scripts.
- Run the demo from a clean environment.
- Save the final metrics and screenshots in the PDF report.
- Keep the limitation about the product-model performance in the discussion section.
