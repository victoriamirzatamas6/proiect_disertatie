from pathlib import Path
import pandas as pd

def cmapss_columns():
    cols = ["unit_id", "cycle", "op1", "op2", "op3"]
    cols += [f"s{i}" for i in range(1, 22)]
    return cols

def load_cmapss(raw_dir: str, fd: str = "FD001"):
    raw_dir = Path(raw_dir)
    train_path = raw_dir / f"train_{fd}.txt"
    test_path  = raw_dir / f"test_{fd}.txt"
    rul_path   = raw_dir / f"RUL_{fd}.txt"

    cols = cmapss_columns()
    train = pd.read_csv(train_path, sep=r"\s+", header=None).iloc[:, :len(cols)]
    test  = pd.read_csv(test_path,  sep=r"\s+", header=None).iloc[:, :len(cols)]
    train.columns = cols
    test.columns  = cols
    rul_test = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL_last"])
    return train, test, rul_test
