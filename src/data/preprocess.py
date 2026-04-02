import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def drop_constant_columns(df_train: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if df_train[c].nunique(dropna=True) > 1]

def fit_scaler(df_train: pd.DataFrame, cols: list[str], kind: str = "standard"):
    scaler = StandardScaler() if kind == "standard" else MinMaxScaler()
    scaler.fit(df_train[cols].values)
    return scaler

def apply_scaler(df: pd.DataFrame, cols: list[str], scaler) -> pd.DataFrame:
    out = df.copy()
    out[cols] = scaler.transform(out[cols].values)
    return out
