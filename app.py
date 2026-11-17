"""
Streamlit web app for the Human Activity Recognition project.

Run:  streamlit run app.py      (opens at http://localhost:8501)
Tabs: live prediction, data & preprocessing charts, model results.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "har_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
FIG = ROOT / "outputs" / "figures"
SAMPLE_PATH = ROOT / "data" / "sample_test.csv"  # 300 test windows shipped with the app
N_FEATURES = 561


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_split():
    # column 0 = true label (1..6), columns 1..561 = features
    df = pd.read_csv(SAMPLE_PATH, header=None)
    return df.iloc[:, 1:].values, df.iloc[:, 0].astype(int).values


def predict(bundle, X):
    model, labels = bundle["model"], bundle["labels"]
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(axis=1)]
    names = [labels[c - 1] for c in pred]  # classes are 1..6
    return names, proba


def figure(name, caption):
    path = FIG / name
    if path.exists():
        st.image(str(path))
        st.caption(caption)
    else:
        st.info(f"{name} not found. Run notebooks/HAR_project.ipynb to generate the figures.")


st.set_page_config(page_title="Human Activity Recognition", page_icon="🏃", layout="wide")
st.title("Human Activity Recognition")
st.write(
    "Predicts what a person is doing (walking, climbing stairs, sitting, "
    "standing, laying) from 561 features extracted from smartphone "
    "accelerometer and gyroscope signals. Dataset: UCI HAR, 30 people."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

bundle = load_model()
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None

if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", bundle["model_name"])
    c2.metric("Test accuracy (unseen people)", f"{metrics['test_accuracy']:.2%}")
    c3.metric("Test macro-F1", f"{metrics['test_macro_f1']:.2%}")
    c4.metric("Train / test windows", "7,352 / 2,947")

tab_predict, tab_data, tab_results = st.tabs(
    ["Try the model", "Data and preprocessing", "Model results"]
)

# ------------------------------------------------------------------ predict
with tab_predict:
    mode = st.radio(
        "Choose input",
        ["Sample from the UCI HAR test set (300 windows)", "Upload a CSV of features"],
        horizontal=True,
    )

    if mode.startswith("Sample"):
        if not SAMPLE_PATH.exists():
            st.error("data/sample_test.csv is missing.")
            st.stop()
        X_test, y_test = load_test_split()
        idx = st.slider("Test window index", 0, len(X_test) - 1, 0)
        names, proba = predict(bundle, X_test[[idx]])
        true_label = bundle["labels"][y_test[idx] - 1]
        left, right = st.columns([1, 2])
        with left:
            st.subheader(f"Predicted: {names[0]}")
            st.write(f"True label (from dataset): **{true_label}**")
            if names[0] == true_label:
                st.success("Correct")
            else:
                st.error("Wrong prediction")
        with right:
            chart_df = pd.DataFrame({"activity": bundle["labels"], "probability": proba[0]})
            bars = (
                alt.Chart(chart_df)
                .mark_bar(color="#2a78d6")
                .encode(
                    x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%"), title="model confidence"),
                    y=alt.Y("activity:N", sort=bundle["labels"], title=None,
                            axis=alt.Axis(labelLimit=220)),
                    tooltip=["activity", alt.Tooltip("probability:Q", format=".1%")],
                )
                .properties(height=260)
            )
            st.altair_chart(bars)
    else:
        st.write(
            f"Upload a CSV with **{N_FEATURES} columns** and **one row per window**, "
            "no header, values already normalised to [-1, 1]."
        )
        file = st.file_uploader("Feature CSV", type="csv")
        if file is not None:
            df = pd.read_csv(file, header=None)
            if df.shape[1] != N_FEATURES:
                st.error(f"Expected {N_FEATURES} columns, got {df.shape[1]}.")
            else:
                names, proba = predict(bundle, df.values.astype(float))
                out = pd.DataFrame({"row": np.arange(len(names)), "prediction": names,
                                    "confidence": proba.max(axis=1).round(3)})
                st.dataframe(out)

# ------------------------------------------------------- data/preprocessing
with tab_data:
    st.subheader("Preprocessing")
    st.markdown(
        """
- **Already done by the dataset authors:** noise filtering, cutting signals into 2.56 s windows (128 readings, 50% overlap), separating body motion from gravity, and computing 561 features per window.
- **Checks done in this project:** 0 missing values, 0 duplicate rows, all values within [-1, 1]; 84 repeated feature names made unique; train (21 people) and test (9 people) share no subjects.
- **Scaling:** `StandardScaler` inside the model pipeline, fitted on training data only, so nothing leaks from validation or test data.
"""
    )
    figure("01_scaling_before_after.png", "Five example features before and after standard scaling.")
    st.subheader("Exploratory data analysis")
    a, b = st.columns(2)
    with a:
        figure("02_class_distribution.png", "Classes are roughly balanced, so accuracy is a fair metric.")
    with b:
        figure("03_windows_per_subject.png", "Each person contributes a similar number of windows.")
    figure("04_raw_signals_per_activity.png",
           "One window of raw body acceleration per activity. Moving activities are rhythmic; still ones are flat.")
    figure("05_key_features_by_activity.png",
           "Movement magnitude separates moving from still; gravity features single out LAYING.")
    a, b = st.columns([1, 2])
    with a:
        figure("06_correlation_heatmap.png", "Many features are strongly correlated (redundant).")
    with b:
        figure("07_pca.png", "PCA: 102 components keep 95% of the variance. SITTING and STANDING overlap.")

# ------------------------------------------------------------------ results
with tab_results:
    st.subheader("Model comparison")
    st.write("5-fold cross-validation grouped by subject, on the training people only.")
    a, b = st.columns(2)
    with a:
        figure("08_model_comparison.png", "Logistic Regression scored highest and was selected.")
    with b:
        if metrics:
            cv = pd.DataFrame(metrics["cv_results"]).T.rename(
                columns={"cv_accuracy_mean": "CV accuracy", "cv_accuracy_std": "std"})
            st.dataframe(cv.style.format("{:.4f}"))
    st.subheader("Final test on 9 unseen people")
    figure("09_confusion_matrices.png", "Most errors: SITTING predicted as STANDING (58 windows).")
    a, b = st.columns(2)
    with a:
        figure("10_per_class_metrics.png", "Precision, recall and F1 for each activity.")
    with b:
        figure("11_feature_importance.png", "Gravity and orientation features carry the most information.")
