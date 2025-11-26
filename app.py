import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/rf_model.pkl")

st.title("F1 Race Position Predictor")
st.write("Enter race details to predict finishing position:")

driver_id = st.number_input("Driver ID", value=1)
constructor_id = st.number_input("Constructor ID", value=1)
grid_position = st.number_input("Grid Position", value=10)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "driverId": driver_id,
        "constructorId": constructor_id,
        "grid": grid_position
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted finishing position: **{prediction:2f}**")