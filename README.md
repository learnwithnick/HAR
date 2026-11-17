# Human Activity Recognition (UCI HAR)

Classifies six activities from smartphone inertial sensors: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.

## Results (official split: 9 unseen test subjects)

| Model | 5-fold grouped CV accuracy (train subjects) |
|---|---|
| **Logistic Regression (selected)** | **0.9345 ± 0.0391** |
| SVM (RBF) | 0.9241 ± 0.0374 |
| Random Forest | 0.9173 ± 0.0439 |

Selected model on the test set: **accuracy 95.52%, macro-F1 95.51%** (`outputs/metrics.json`, `outputs/confusion_matrix.png`).

## Method

- Data: UCI HAR Dataset, 561 pre-extracted features per 2.56 s window, 21 train / 9 test subjects.
- No leakage: model selection uses `GroupKFold` by subject, so a person's windows never appear in both training and validation folds. The test set is used once, after selection.
- Scaling is inside the sklearn pipeline, so it is fitted on training folds only.
- `random_state=42` everywhere for reproducibility.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `UCI HAR Dataset.zip` from the UCI Machine Learning Repository (dataset 240) and unzip it so the folder is `data/UCI HAR Dataset/`.

## Train

```bash
python train.py --data "data/UCI HAR Dataset" --out .
```

Creates `models/har_model.joblib`, `outputs/metrics.json`, and `outputs/confusion_matrix.png`. Takes about 2 minutes on a laptop.

## Run the web app (Streamlit)

```bash
streamlit run app.py
```

Opens at http://localhost:8501. You can browse test windows or upload a CSV with 561 columns and one row per window.

## Deploy

Streamlit Community Cloud: push this folder to GitHub (include `models/` and `data/` or a small test sample), then deploy `app.py` from share.streamlit.io. Alternatively, run `streamlit run app.py --server.port 8501` on any Linux VM or container.
