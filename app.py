import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

drivers_df = pd.read_csv("data/drivers.csv")
constructors_df = pd.read_csv("data/constructors.csv")

driver_options = dict(zip(drivers_df["surname"], drivers_df['driverId']))
constructor_options = dict(zip(constructors_df["name"], constructors_df["constructorId"]))

st.set_page_config(page_title="F1 Race Position Predictor", layout="centered")

st.title("F1 Race Position Predictor!")
st.markdown(
    """
    Welcome to the **F1 Machine Learning Prediction App**!
    Select a **driver**, **constructor**, and **grid position**, and the model will predict the finishing position.
    """
)

st.divider()

model_choice = st.selectbox(
    "Choose ML Model:",
    ["Random Forest", "XGBoost"]
)

driver_name = st.selectbox("Driver:", list(driver_options.keys()))
constructor_name = st.selectbox("Constructor:", list(constructor_options.keys()))
grid_position = st.slider("Grid Position (Starting Position)", 1, 20, 10)

st.divider()

if st.button("Predict Finishing Position"):

    chosen_driver = driver_options[driver_name]
    chosen_constructor = constructor_options[constructor_name]

    model = rf_model if model_choice == "Random Forest" else xgb_model

    input_dict = {feat:0 for feat in model.feature_names_in_}

    input_data = pd.DataFrame({
        "driverId": chosen_driver,
        "constructorId": chosen_constructor,
        "grid": grid_position
    })

    input_data = pd.DataFrame([input_dict])

    prediction = model.predict(input_data)[0]

    st.success(f"**Predicted finishing position: `{prediction: .2f}`**")

    if prediction < grid_position:
        st.markdown("**The model predicts the driver will finish above the starting position.**")
    elif prediction > grid_position:
        st.markdown("**The model predicts the driver may drop positions in the race.**")
    else:
        st.markdown("**The model predicts the driver will finish in the same spot they started.**")

    st.divider()


