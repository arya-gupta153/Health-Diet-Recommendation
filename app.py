import streamlit as st
import joblib
import numpy as np

# Load model + encoders
model = joblib.load("notebook/model.pkl")
le_severity = joblib.load("notebook/le_severity.pkl")
le_activity = joblib.load("notebook/le_activity.pkl")
le_disease = joblib.load("notebook/le_disease.pkl")

# Diet recommendation logic
def recommend_diet(disease):
    if disease == "Diabetes":
        return "Low sugar diet, whole grains, vegetables, lean protein"
    elif disease == "Obesity":
        return "Low calorie diet, high fiber, avoid junk food"
    elif disease == "Hypertension":
        return "Low salt diet, fruits, vegetables, DASH diet"
    else:
        return "Balanced diet with proteins, carbs, fats"

# UI
st.title("🥗 Diet Recommendation System")

weight = st.number_input("Weight (kg)")
height = st.number_input("Height (cm)")
bmi = st.number_input("BMI")

severity = st.selectbox("Severity", le_severity.classes_)
activity = st.selectbox("Physical Activity", le_activity.classes_)

if st.button("Predict Diet"):
    # Encode input safely
    severity_enc = le_severity.transform([severity])[0]
    activity_enc = le_activity.transform([activity])[0]

    # Predict disease
    input_data = np.array([[weight, height, bmi, severity_enc, activity_enc]])
    pred = model.predict(input_data)

    disease = le_disease.inverse_transform(pred)[0]

    # Get diet
    diet = recommend_diet(disease)

    st.success(f"Predicted Disease: {disease}")
    st.info(f"Recommended Diet: {diet}")