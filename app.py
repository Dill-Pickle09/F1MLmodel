import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

df_drivers = pd.read_csv("data/drivers.csv")
df_constructors = pd.read_csv("data/constructors.csv")
df_results = pd.read_csv("data/results.csv")
df_races = pd.read_csv("data/races.csv")


df = df_results.merge(df_drivers, on="driverId", how="left") \
               .merge(df_constructors, on="constructorId", how="left") \
               .merge(df_races, on="raceId", how="left")


driver_options = dict(zip(df_drivers["surname"], df_drivers['driverId']))
constructor_options = dict(zip(df_constructors["name"], df_constructors["constructorId"]))

st.set_page_config(page_title="F1 Race Position Predictor", layout="centered")
st.title("F1 Race Position Predictor!")
st.markdown(
    """
    Welcome to the **F1 Machine Learning Prediction App**!
    Select a **driver**, **constructor**, and **grid position**, and the model will predict the finishing position.
    """
)

st.divider()

driver_name = st.selectbox("Driver:", list(driver_options.keys()))
constructor_name = st.selectbox("Constructor:", list(constructor_options.keys()))
grid_position = st.slider("Grid Position", 1, 20, 10)
year = st.number_input("Year", 2000, 2025, 2020)
round_ = st.slider("Round", 1, 23, 1)
model_choice = st.selectbox("Choose ML Model:", ["Random Forest", "XGBoost"])

st.divider()

if st.button("Predict Finishing Position"):
    driver_id = driver_options[driver_name]
    constructor_id = constructor_options[constructor_name]

    df_past = df[(df["year"] < year) | ((df["year"] == year) & (df["round"] < round_))]

    driver_past = df_past[df_past["driverId"] == driver_id].sort_values(["year", "round"], ascending=False)
    last_5 = driver_past.head(5)
    driver_avg_finish_last_5 = last_5["positionOrder"].mean() if not last_5.empty else 10
    driver_points_last_5 = last_5["points"].sum() if not last_5.empty else 5

    df_last_race = df_past[(df_past["constructorId"] == constructor_id) & (df_past["driverId"] != driver_id)]
    if not df_last_race.empty:
        teammate = df_last_race.sort_values(["year", "round"], ascending=False).iloc[0]
        grid_diff_teammate = grid_position - teammate["grid"]
        finish_diff_teammate = 0
        points_diff_teammate = 0
    else:
        grid_diff_teammate = 0
        finish_diff_teammate = 0
        points_diff_teammate = 0
    
    current_circuit = df_races[(df_races["year"] == year) & (df_races["round"] == round_)]
    if not current_circuit.empty:
        circuit_id = current_circuit.iloc[0]["circuitId"]
        driver_circuit = df_past[(df_past["driverId"] == driver_id) & (df_past["circuitId"] == circuit_id)]
    else:
        driver_circuit = pd.DataFrame()



    driver_circuit_avg_finish = driver_circuit["positionOrder"].mean() if not driver_circuit.empty else driver_avg_finish_last_5

    input_dict = {
        "year": year,
        "round": round_,
        "grid": grid_position,
        "driver_avg_finish_last_5": driver_avg_finish_last_5,
        "driver_points_last_5":driver_points_last_5,
        "grid_diff_teammate": grid_diff_teammate,
        "finish_diff_teammate": finish_diff_teammate,
        "points_diff_teammate": points_diff_teammate,
        "driver_circuit_avg_finish": driver_circuit_avg_finish
    }

    model = rf_model if model_choice == "Random Forest" else xgb_model
    for feat in model.feature_names_in_:
        if feat not in input_dict:
            input_dict[feat] = 0

    input_data = pd.DataFrame([input_dict])[model.feature_names_in_]

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted finishing position: *{prediction: .2f}**")

    if prediction < grid_position:
        st.markdown("The driver is predicted to **gain poisitions** during the race")
    elif prediction > grid_position:
        st.markdown("The driver is predicted to **drop positions** during the race ")
    else:
        st.markdown("The driver is predicted to finish where they started")




    st.divider()


