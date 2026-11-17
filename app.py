"""
Streamlit web app for the Human Activity Recognition model.

Run:  streamlit run app.py
Opens at http://localhost:8501
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "har_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
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


st.set_page_config(page_title="Human Activity Recognition", page_icon="🏃")
st.title("Human Activity Recognition")
st.write(
    "Predicts what a person is doing (walking, climbing stairs, sitting, "
    "standing, laying) from the 561 features extracted from smartphone "
    "accelerometer and gyroscope signals."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

bundle = load_model()
st.caption(f"Model: {bundle['model_name']}")

with st.sidebar:
    st.header("Model performance")
    if METRICS_PATH.exists():
        import json

        m = json.loads(METRICS_PATH.read_text())
        st.metric("Test accuracy (unseen people)", f"{m['test_accuracy']:.2%}")
        st.metric("Test macro-F1", f"{m['test_macro_f1']:.2%}")
        st.image(str(ROOT / "outputs" / "confusion_matrix.png"))
    else:
        st.info("Run train.py to generate metrics.")

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
    st.subheader(f"Predicted: {names[0]}")
    st.write(f"True label (from dataset): **{true_label}**")
    result = pd.DataFrame(proba, columns=bundle["labels"]).T.rename(columns={0: "probability"})
    st.bar_chart(result)

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
