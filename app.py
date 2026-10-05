import streamlit as st
import joblib
import pandas as pd

from lexical_features import extract_lexical_features

model = joblib.load("models/final_model.joblib")
feature_cols = joblib.load("models/feature_columns.joblib")


def predict_url(url: str) -> dict:
    features = extract_lexical_features(pd.DataFrame({"url": [url]}), url_col="url")
    X_new = features[feature_cols]

    proba = model.predict_proba(X_new)[0, 1]
    trust_score = round((1 - proba) * 10, 2)

    return {
        "phishing_probability": float(round(proba, 4)),
        "trust_score": float(trust_score),
        "verdict": "PHISHING" if proba >= 0.5 else "LEGITIMATE",
        "features": features.iloc[0].to_dict(),
    }


st.set_page_config(page_title="Phishing URL Detector", layout="centered")

st.markdown("<h1 style='text-align: center; color: #00C9FF;'>Phishing URL Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AAB5;'>Paste a URL below to check if it is a phishing attempt.</p>", unsafe_allow_html=True)

url_input = st.text_input("Enter a URL:", placeholder="http://example.com")

if st.button("Analyze URL", use_container_width=True):
    if url_input:
        result = predict_url(url_input)
        st.markdown("---")
        if result["verdict"] == "PHISHING":
            st.error("PHISHING - This URL looks suspicious!")
        else:
            st.success("LEGITIMATE - This URL appears safe.")
        col1, col2 = st.columns(2)
        col1.metric("Trust Score (0-10)", result["trust_score"])
        col2.metric("Phishing Probability", str(round(result["phishing_probability"] * 100, 2)) + "%")
        with st.expander("View Feature Breakdown"):
            st.json(result["features"])
    else:
        st.warning("Please enter a URL first.")
