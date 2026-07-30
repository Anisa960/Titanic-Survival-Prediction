import streamlit as st
import pandas as pd
import joblib
import sklearn
st.write("scikit-learn:", sklearn.__version__)


st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Provide passenger details to predict survival. The app expects a trained `model.pkl`.")

@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error("Could not load model.pkl. Run the notebook first to train & save the model.")
        st.stop()

model = load_model()

with st.form("form"):
    pclass = st.selectbox("Pclass", [1,2,3], index=2)
    sex = st.selectbox("Sex", ["male", "female"], index=0)
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
    sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=0.1)
    embarked = st.selectbox("Embarked", ["C","Q","S"], index=2)
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([{
        "Pclass": str(pclass), "Sex": sex, "Age": age, "SibSp": sibsp,
        "Parch": parch, "Fare": fare, "Embarked": embarked
    }])
    proba = model.predict_proba(X)[0,1]
    pred = "Survived" if proba >= 0.5 else "Not Survived"
    st.subheader(f"Prediction: {pred}")
    st.write(f"Survival probability: {proba:.2%}")
st.write(type(model))
print(model)
from pprint import pprint
pprint(model)