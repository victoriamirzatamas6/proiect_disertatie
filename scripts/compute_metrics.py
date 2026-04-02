import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# === Alege modelul ===
csv_path = "outputs/predictions/rul_baseline_xgb.csv"
model_name = "XGBoost"

df = pd.read_csv(csv_path)

y_true = df["RUL"].values
y_pred = df["pred_RUL_xgb"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Model: {model_name}")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")