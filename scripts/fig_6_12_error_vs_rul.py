from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # === Alege modelul aici ===
    # LSTM:
    csv_path = Path("outputs/predictions/rul_lstm.csv")
    model_name = "LSTM"

    # Dacă vrei XGBoost, comentează cele de mai sus și folosește:
    # csv_path = Path("outputs/predictions/rul_baseline_xgb.csv")
    # model_name = "XGBoost"

    if not csv_path.exists():
        raise FileNotFoundError(f"Nu găsesc fișierul: {csv_path}")

    df = pd.read_csv(csv_path)

    if "RUL" not in df.columns:
        raise ValueError(f"Nu există coloana RUL în {csv_path}. Coloane: {list(df.columns)}")

    # Detectează automat coloana de predicție (prima care conține 'pred')
    pred_cols = [c for c in df.columns if "pred" in c.lower()]
    if not pred_cols:
        raise ValueError(f"Nu găsesc coloană de predicție în {csv_path}. Coloane: {list(df.columns)}")
    pred_col = pred_cols[0]

    y_true = df["RUL"].values.astype(float)
    y_pred = df[pred_col].values.astype(float)

    abs_error = np.abs(y_true - y_pred)

    out_dir = Path("outputs/figures/ch6")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fig_6_12_error_vs_rul_{model_name.lower()}.png"

    # Stil disertație
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(y_true, abs_error, alpha=0.45)

    ax.set_xlabel("RUL real (cicluri)")
    ax.set_ylabel("Eroare absolută (cicluri)")
    ax.set_title(f"Eroare absolută în funcție de RUL – {model_name} (test)", pad=10)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")
    print(f"[INFO] Using prediction column: {pred_col}")


if __name__ == "__main__":
    main()