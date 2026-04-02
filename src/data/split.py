import numpy as np
import pandas as pd

def split_by_units(df: pd.DataFrame, train_units_ratio: float, seed: int):
    units = df["unit_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_train = int(len(units) * train_units_ratio)
    train_units = set(units[:n_train])
    valid_units = set(units[n_train:])
    return df[df["unit_id"].isin(train_units)].copy(), df[df["unit_id"].isin(valid_units)].copy(), sorted(train_units), sorted(valid_units)
