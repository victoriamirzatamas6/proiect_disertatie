import numpy as np
import pandas as pd

def _slope(x: np.ndarray) -> float:
    t = np.arange(len(x))
    if np.std(x) < 1e-12:
        return 0.0
    return float(np.polyfit(t, x, 1)[0])

def make_tabular_window_features(df: pd.DataFrame, feature_cols: list[str], window: int, step: int, cap_rul: int | None):
    rows = []
    for unit_id, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        X = g[feature_cols].values
        y = g["RUL"].values
        cycles = g["cycle"].values
        for end in range(window, len(g) + 1, step):
            seg = X[end-window:end, :]
            feats = {}
            for j, col in enumerate(feature_cols):
                x = seg[:, j]
                feats[f"{col}_mean"] = float(np.mean(x))
                feats[f"{col}_std"]  = float(np.std(x))
                feats[f"{col}_min"]  = float(np.min(x))
                feats[f"{col}_max"]  = float(np.max(x))
                feats[f"{col}_slope"]= _slope(x)
            target = float(y[end - 1])
            if cap_rul is not None:
                target = float(min(target, cap_rul))
            rows.append({"unit_id": int(unit_id), "cycle_end": int(cycles[end-1]), **feats, "RUL": target})
    return pd.DataFrame(rows)

def last_window_per_unit(df_windows: pd.DataFrame, unit_col: str = "unit_id", time_col: str = "cycle_end"):
    idx = df_windows.groupby(unit_col)[time_col].idxmax()
    return df_windows.loc[idx].sort_values(unit_col).reset_index(drop=True)
