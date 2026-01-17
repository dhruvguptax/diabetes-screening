import streamlit as st
import pandas as pd
import pickle


with open("model/diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)


st.set_page_config(page_title="Diabetes Risk Screening AI")


st.title("Diabetes Risk Screening AI")
st.write("Enter basic health details to estimate diabetes risk.")
st.warning("⚠️ This is a screening tool, not a medical diagnosis.")


pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
)
age = st.number_input("Age", min_value=1, max_value=120, value=30)


if st.button("Check Diabetes Risk"):
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigree",
            "Age"
        ]
    )

    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    low_risk_prob = probability[0][0] * 100
    high_risk_prob = probability[0][1] * 100

    st.write("Risk Assessment Result")

    if prediction[0] == 1:
        st.error("High risk of diabetes")
        st.write(f"Estimated Risk: {high_risk_prob:.2f}%")
    else:
        st.success("Low risk of diabetes")
        st.write(f"Estimated Risk: {high_risk_prob:.2f}%")

    st.info(
        "The percentage shown represents the model’s confidence based on learned patterns "
        "from the training data. It is not a medical diagnosis."
    )
