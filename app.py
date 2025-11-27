import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

MODELS_DIR = "models"
meta = joblib.load(os.path.join(MODELS_DIR, "feature_metadata.pkl"))
feature_list = meta["feature_list"]
means = meta["means"]

rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))

DATA_DIR = "data"
df_results = pd.read_csv(f"{DATA_DIR}/results.csv")
df_drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
df_constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
df_races = pd.read_csv(f"{DATA_DIR}/races.csv")

df = (
    df_results.merge(df_drivers, on="driverId", how="left")
              .merge(df_constructors, on="constructorId", how="left")
              .merge(df_races, on="raceId", how="left")
)

driver_options = dict(zip(df_drivers["surname"], df_drivers['driverId']))
constructor_options = dict(zip(df_constructors["name"], df_constructors['constructorId']))




st.set_page_config(page_title="F1 Race Result Predictor", layout="centered")
st.title("F1 Race Finishing Position Predictor")
st.markdown("Select a driver, constructor, starting grid slot, year, and round below.")

driver_name = st.selectbox("Driver", list(driver_options.keys()))
constructor_name = st.selectbox("Constructor", list(constructor_options.keys()))
grid_position = st.slider("Grid Position", 1, 20, 10)
year = st.number_input("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].min()))
round_ = st.slider("Round", 1, int(df['round'].max()), 1)
model_choice = st.selectbox("Model", ["Random Forest","XGBoost"])

st.divider()

def compute_features_for_case(driver_id, constructor_id, year, round_, grid):
    df_past = df[(df["year"] < year) | ((df["year"] == year) & (df["round"] < round_))]

    driver_past = df_past[df_past["driverId"] == driver_id].sort_values(["year", "round"], ascending=False)
    last_5 = driver_past.head(5)
    driver_avg_finish_last_5 = last_5["positionOrder"].mean() if not last_5.empty else means.get('driver_avg_finish_last_5', 10)
    driver_points_last_5 = last_5["points"].sum() if not last_5.empty else means.get('driver_points_last_5', 5)

    df_last_race = df_past[(df_past["constructorId"] == constructor_id) & (df_past["driverId"] != driver_id)]
    if not df_last_race.empty:
        teammate = df_last_race.sort_values(["year","round"], ascending=False).iloc[0]
        grid_diff_teammate = grid - teammate["grid"]
        finish_diff_teammate = 0 if pd.isna(teammate["positionOrder"]) else 0 - teammate["positionOrder"]  
        points_diff_teammate = 0 if pd.isna(teammate["points"]) else 0 - teammate["points"]
    else:
        grid_diff_teammate = 0
        finish_diff_teammate = 0
        points_diff_teammate = 0


    current_race = df_races[(df_races["year"] == year) & (df_races["round"] == round_)]
    if not current_race.empty:
        circuit_id = current_race.iloc[0]["circuitId"]
        driver_circuit = df_past[(df_past["driverId"] == driver_id) & (df_past["circuitId"] == circuit_id)]
    else:
        driver_circuit = df_past[df_past["driverId"] == driver_id]

    driver_circuit_avg_finish = driver_circuit["positionOrder"].mean() if not driver_circuit.empty else driver_avg_finish_last_5


    driver_season = df_past[(df_past["driverId"] == driver_id) & (df_past["year"] == year)]
    driver_season_points_to_date = driver_season["points"].sum() if not driver_season.empty else 0

    if not driver_season.empty:
        driver_season_avg_finish_to_date = driver_season["positionOrder"].mean()
    else:
        driver_season_avg_finish_to_date = driver_avg_finish_last_5

    driver_total = df_results[df_results['driverId'] == driver_id].shape[0]
    driver_dnfs = df_results[(df_results['driverId'] == driver_id) & (df_results['positionOrder'].isna())].shape[0]
    driver_dnf_rate = driver_dnfs / driver_total if driver_total>0 else 0

    constructor_past = df_past[df_past["constructorId"] == constructor_id].sort_values(["year", "round"], ascending=False)
    c_last5 = constructor_past.head(5)
    constructor_avg_finish_last_5 = c_last5["positionOrder"].mean() if not c_last5.empty else means.get('constructor_avg_finish_last_5', driver_avg_finish_last_5)
    constructor_points_last_5 = c_last5["points"].sum() if not c_last5.empty else means.get('constructor_points_last_5', 0)

    c_total = df_results[df_results['constructorId']==constructor_id].shape[0]
    c_dnfs = df_results[(df_results['constructorId']==constructor_id) & (df_results['positionOrder'].isna())].shape[0]
    constructor_dnf_rate = c_dnfs / c_total if c_total>0 else 0

    race_row = df_races[(df_races["year"]==year) & (df_races["round"]==round_)]
    is_sprint = 1 if (not race_row.empty and pd.notna(race_row.iloc[0].get('sprint_date'))) else 0

    d = {
        'year': int(year),
        'round': int(round_),
        'grid': int(grid),
        'driver_avg_finish_last_5': float(driver_avg_finish_last_5),
        'driver_points_last_5': float(driver_points_last_5),
        'grid_diff_teammate': float(grid_diff_teammate),
        'finish_diff_teammate': float(finish_diff_teammate),
        'points_diff_teammate': float(points_diff_teammate),
        'driver_circuit_avg_finish': float(driver_circuit_avg_finish),
        'driver_season_points_to_date': float(driver_season_points_to_date),
        'driver_season_avg_finish_to_date': float(driver_season_avg_finish_to_date),
        'driver_dnf_rate': float(driver_dnf_rate),
        'constructor_avg_finish_last_5': float(constructor_avg_finish_last_5),
        'constructor_points_last_5': float(constructor_points_last_5),
        'constructor_dnf_rate': float(constructor_dnf_rate),
        'is_sprint': int(is_sprint)
    }

    for feat in feature_list:
        if feat not in d:
            d[feat] = float(means.get(feat, 0))

    return d

if st.button("Predict"):
    driver_id = driver_options[driver_name]
    constructor_id = constructor_options[constructor_name]

    feat_dict = compute_features_for_case(driver_id, constructor_id, year, round_, grid_position)
    input_df = pd.DataFrame([feat_dict])[feature_list]

    model = rf_model if model_choice == "Random Forest" else xgb_model
    pred = model.predict(input_df)[0]

    st.success(f"Predicted finishing position: **{pred:.2f}**")
    if pred < grid_position:
        st.markdown("Model predicts the driver will gain positions")
    elif pred > grid_position:
        st.markdown("Model predicts the driver will lose positions")
    else:
        st.markdown("Model predicts the driver will not gain or lose places")

    feat_dict_clean = {k: (int(v) if isinstance(v, (np.integer)) else float(v) if isinstance(v, (np.floating)) else v) for k,v in feat_dict.items()}
    st.json(feat_dict_clean)

    st.divider