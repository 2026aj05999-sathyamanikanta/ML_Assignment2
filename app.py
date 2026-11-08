import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score,
    recall_score, 
    f1_score, 
    matthews_corrcoef, 
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🧬", layout="wide")
st.title("Breast Cancer Classification")
st.caption("Educational comparison of five classifiers. Not a medical diagnostic tool.")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib"
}

uploaded = st.file_uploader("Upload labelled test CSV", type=["csv"])
selected = st.selectbox("Select model", list(model_files))

if uploaded is not None:
    data = pd.read_csv(uploaded)
    if "diagnosis" not in data:
        st.error("CSV must contain diagnosis target column.")
        st.stop()
    raw_y = data["diagnosis"]
    if raw_y.dtype == object:
        y = raw_y.astype(str).str.strip().str.upper().map({"B":0, "M":1, "0":0, "1":1})
    else:
        y = pd.to_numeric(raw_y, errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0,1}):
        st.error("Diagnosis must contain B/M or 0/1.")
        st.stop()
    y = y.astype(int)
    X = data.drop(columns=["diagnosis", "id", "Unnamed: 32"], errors="ignore")
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except Exception as e:
        st.error(f"Features must be numeric: {e}")
        st.stop()
    if X.isna().any().any():
        st.error("Features contain missing values.")
        st.stop()
    
    path = os.path.join("models", model_files[selected])
    if not os.path.exists(path):
        st.error(f"Model file missing: {path}")
        st.stop()
    model = joblib.load(path)
    expected = list(model.feature_names_in_)
    missing, extra = sorted(set(expected)-set(X.columns)), sorted(set(X.columns)-set(expected))
    if missing or extra:
        st.error(f"Feature mismatch. Missing: {missing}; Extra: {extra}")
        st.stop()
    X = X[expected]
    pred = model.predict(X)
    prob = model.predict_proba(X)[:,1]
    if len(np.unique(y)) < 2:
        st.error("AUC requires both classes in uploaded test data.")
        st.stop()
    
    st.subheader("Uploaded data")
    st.dataframe(data.head())
    
    vals = {
        "Accuracy": accuracy_score(y,pred),
        "AUC": roc_auc_score(y,prob),
        "Precision": precision_score(y,pred,zero_division=0),
        "Recall": recall_score(y,pred,zero_division=0),
        "F1": f1_score(y,pred,zero_division=0),
        "MCC": matthews_corrcoef(y,pred)
    }
    
    st.subheader("Evaluation metrics")
    cols = st.columns(3)
    for i,(k,v) in enumerate(vals.items()):
        cols[i%3].metric(k, f"{v:.4f}")
    
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y,pred,labels=[0,1]),
        index=["Actual Benign","Actual Malignant"],
        columns=["Predicted Benign","Predicted Malignant"]))
    
    st.subheader("Classification report")
    rep = classification_report(y,pred,labels=[0,1],
        target_names=["Benign","Malignant"],output_dict=True,zero_division=0)
    st.dataframe(pd.DataFrame(rep).transpose())
else:
    st.info("Upload a labelled CSV with all 30 predictors and diagnosis column.")
