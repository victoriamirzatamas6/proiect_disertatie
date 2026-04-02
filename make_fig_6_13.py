import pandas as pd
import matplotlib.pyplot as plt

# citește rezultatele ablației
df = pd.read_csv("outputs/metrics/ablation_results.csv")

# selectează doar LSTM
df_lstm = df[df["model"] == "lstm"]

# sortează după dimensiunea ferestrei
df_lstm = df_lstm.sort_values("window")

windows = df_lstm["window"]
mae = df_lstm["mae"]
rmse = df_lstm["rmse"]

plt.figure(figsize=(7,5))

plt.plot(windows, mae, marker='o', label="MAE")
plt.plot(windows, rmse, marker='o', label="RMSE")

plt.xlabel("Dimensiunea ferestrei (cicluri)")
plt.ylabel("Eroare")
plt.title("Performanța modelului LSTM în funcție de dimensiunea ferestrei")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/figures/fig_6_13_window_sensitivity.png", dpi=300)

plt.show()