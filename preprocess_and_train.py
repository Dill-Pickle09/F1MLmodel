import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

df_results = pd.read_csv(f"{DATA_DIR}/results.csv")
df_drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
df_constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
df_races = pd.read_csv(f"{DATA_DIR}/races.csv")

df = (
    df_results.merge(df_drivers, on="driverId", how="left")
              .merge(df_constructors, on="constructorId", how="left")
              .merge(df_races, on="raceId", how="left")
)

df['name_driver'] = df['forename'].fillna('') + ' ' + df['surname'].fillna('')
df_basic = df[[
    'raceId','year','round','driverId','constructorId','grid','positionOrder','points','circuitId'
]].copy()
df_basic = df_basic[df_basic['positionOrder'].notna()]

df_basic = df_basic.sort_values(by=['driverId', 'year', 'round']).reset_index(drop=True)

df_basic['driver_avg_finish_last_5'] = (
    df_basic.groupby('driverId')['positionOrder']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
)

df_basic['driver_points_last_5'] = (
    df_basic.groupby('driverId')['points']
            .rolling(window=5, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
)

df_basic['driver_avg_finish_last_5'] = df_basic['driver_avg_finish_last_5'].fillna(df_basic['driver_avg_finish_last_5'].mean())
df_basic['driver_points_last_5'] = df_basic['driver_points_last_5'].fillna(0)

df_teammate = df_basic[['raceId','constructorId','driverId','grid','positionOrder','points']].copy()

df_basic = df_basic.merge(df_teammate,
                          on=['raceId','constructorId'],
                          suffixes=('','_teammate'))

df_basic = df_basic[df_basic['driverId'] != df_basic['driverId_teammate']]

df_basic['grid_diff_teammate'] = df_basic['grid'] - df_basic['grid_teammate']
df_basic['finish_diff_teammate'] = df_basic['positionOrder'] - df_basic['positionOrder_teammate']
df_basic['points_diff_teammate'] = df_basic['points'] - df_basic['points_teammate']

df_basic = df_basic.drop(columns=['driverId_teammate','grid_teammate','positionOrder_teammate','points_teammate'])


driver_circuit_avg = df_basic.groupby(['driverId','circuitId'])['positionOrder'].mean().reset_index()
driver_circuit_avg.rename(columns={'positionOrder':'driver_circuit_avg_finish'}, inplace=True)
df_basic = df_basic.merge(driver_circuit_avg, on='driverId', how='left')

df_basic['driver_circuit_avg_finish'] = df_basic['driver_circuit_avg_finish'].fillna(
    df_basic['driver_avg_finish_last_5']
)


df_basic = df_basic.sort_values(['driverId','year','round']).reset_index(drop=True)
df_basic['driver_season_points_to_date'] = df_basic.groupby(['driverId','year'])['points'].cumsum() - df_basic['points']

df_basic['driver_season_finishes_count'] = df_basic.groupby(['driverId','year']).cumcount()
df_basic['driver_season_finish_sum'] = df_basic.groupby(['driverId','year'])['positionOrder'].cumsum() - df_basic['positionOrder']
df_basic['driver_season_avg_finish_to_date'] = np.where(
    df_basic['driver_season_finishes_count']>0,
    df_basic['driver_season_finish_sum'] / df_basic['driver_season_finishes_count'],
    np.nan
)
df_basic['driver_season_avg_finish_to_date'] = df_basic['driver_season_avg_finish_to_date'].fillna(df_basic['driver_avg_finish_last_5'])

full_results = df_results.copy()
full_results['dnf'] = full_results['positionOrder'].isna() | (full_results['position'].astype(str).str.upper().isin(['R']))

driver_total = df_results.groupby('driverId').size().rename('total_races').reset_index()
driver_dnfs = df_results[df_results['positionOrder'].isna()].groupby('driverId').size().rename('dnfs').reset_index()
driver_stats = driver_total.merge(driver_dnfs, on='driverId', how='left').fillna(0)
driver_stats['driver_dnf_rate'] = driver_stats['dnfs'] / driver_stats['total_races']

df_basic = df_basic.merge(driver_stats[['driverId','driver_dnf_rate']], on='driverId', how='left')




df_basic = df_basic.sort_values(by=['constructorId','year','round']).reset_index(drop=True)
df_basic['constructor_avg_finish_last_5'] = (
    df_basic.groupby('constructorId')['positionOrder']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
)
df_basic['constructor_points_last_5'] = (
    df_basic.groupby('constructorId')['points']
            .rolling(window=5, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
)
df_basic['constructor_avg_finish_last_5'] = df_basic['constructor_avg_finish_last_5'].fillna(df_basic['constructor_avg_finish_last_5'].mean())
df_basic['constructor_points_last_5'] = df_basic['constructor_points_last_5'].fillna(0)


constructor_total = df_results.groupby('constructorId').size().rename('c_total_races').reset_index()
constructor_dnfs = df_results[df_results['positionOrder'].isna()].groupby('constructorId').size().rename('c_dnfs').reset_index()
constructor_stats = constructor_total.merge(constructor_dnfs, on='constructorId', how='left').fillna(0)
constructor_stats['constructor_dnf_rate'] = constructor_stats['c_dnfs'] / constructor_stats['c_total_races']
df_basic = df_basic.merge(constructor_stats[['constructorId','constructor_dnf_rate']], on='constructorId', how='left')
df_basic['constructor_dnf_rate'] = df_basic['constructor_dnf_rate'].fillna(0)

df_races_small = df_races[['raceId','sprint_date']].copy()
df_races_small['is_sprint'] = df_races_small['sprint_date'].notna()
df_basic = df_basic.merge(df_races_small, on='raceId', how='left')
df_basic['is_sprint'] = df_basic['is_sprint'].fillna(False).astype(int)

feature_list = [
    'year','round','grid',
    'driver_avg_finish_last_5','driver_points_last_5',
    'grid_diff_teammate','finish_diff_teammate','points_diff_teammate','driver_circuit_avg_finish',
    'driver_season_points_to_date','driver_season_avg_finish_to_date','driver_dnf_rate',
    'constructor_avg_finish_last_5','constructor_points_last_5','constructor_dnf_rate',
    'is_sprint'
]


features_df = df_basic[feature_list].copy()
means = features_df.mean(numeric_only=True).to_dict()
features_df = features_df.fillna(means).fillna(0)

target = df_basic['positionOrder'].copy()


max_year = df_basic['year'].max()
train_mask = df_basic['year'] < max_year
test_mask = df_basic['year'] == max_year
X_train = features_df[train_mask.values]
y_train = target[train_mask.values]
X_test = features_df[test_mask.values]
y_test = target[test_mask.values]

if X_test.shape[0] == 0:
    X_train, X_test, y_train, y_test = train_test_split(features_df, target, test_size=0.2, shuffle=False)

rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


def print_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Performance:\n MAE: {mae:.2f}\n RMSE: {rmse:.2f}\n R^2: {r2:.2f}\n")

print_metrics(y_test, y_pred_rf, "Random Forest")
print_metrics(y_test, y_pred_xgb, "XGBoost")

meta = {
    "feature_list": feature_list,
    "means": means
}
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_model.pkl"))
joblib.dump(meta, os.path.join(MODELS_DIR, "feature_metadata.pkl"))

print("Saved models and metadata to", MODELS_DIR)


