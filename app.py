# Web app: paste a URL and the saved model says if it looks like phishing.
#
# Run with:  streamlit run app.py
# (Run train.py first so the model files exist in models/)

import joblib
import pandas as pd
import streamlit as st

from lexical_features import extract_lexical_features


# Load the model only once. Without this, Streamlit
# would reload it every time you click something.
@st.cache_resource
def load_model():
    model = joblib.load("models/final_model.joblib")
    feature_cols = joblib.load("models/feature_columns.joblib")
    return model, feature_cols


model, feature_cols = load_model()


def predict_url(url: str) -> dict:
    # Check one URL with the saved model.
    #
    # It uses the same feature code as training, so the model sees the URL
    # the same way it saw the training data.
    # Trust score: 10 = surely safe, 0 = surely phishing.
    features = extract_lexical_features(pd.DataFrame({"url": [url]}), url_col="url")
    # Put the columns in the same order as during training
    X_new = features[feature_cols]

    # Chance that this URL is phishing (0 to 1)
    proba = model.predict_proba(X_new)[0, 1]
    trust_score = round((1 - proba) * 10, 2)

    return {
        "phishing_probability": float(round(proba, 4)),
        "trust_score": float(trust_score),
        "verdict": "PHISHING" if proba >= 0.5 else "LEGITIMATE",
        "features": features.iloc[0].to_dict(),
    }


# Page layout
st.set_page_config(page_title="Phishing URL Detector", layout="centered")

st.markdown("<h1 style='text-align: center; color: #00C9FF;'>Phishing URL Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AAB5;'>Paste a URL below to check if it is a phishing attempt.</p>", unsafe_allow_html=True)

url_input = st.text_input("Enter a URL:", placeholder="http://example.com").strip()

if st.button("Analyze URL", use_container_width=True):
    if url_input:
        result = predict_url(url_input)
        st.markdown("---")
        if result["verdict"] == "PHISHING":
            st.error("PHISHING - This URL looks suspicious!")
        else:
            st.success("LEGITIMATE - This URL appears safe.")

        # Show two numbers side by side
        col1, col2 = st.columns(2)
        col1.metric("Trust Score (0-10)", result["trust_score"])
        col2.metric("Phishing Probability", str(round(result["phishing_probability"] * 100, 2)) + "%")

        with st.expander("View Feature Breakdown"):
            st.json(result["features"])
    else:
        st.warning("Please enter a URL first.")
