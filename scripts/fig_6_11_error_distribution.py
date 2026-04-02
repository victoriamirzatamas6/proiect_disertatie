from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # === Alege modelul aici ===
    csv_path = Path("outputs/predictions/rul_lstm.csv")
    model_name = "LSTM"   # sau "XGBoost"

    if not csv_path.exists():
        raise FileNotFoundError(f"Nu găsesc fișierul: {csv_path}")

    df = pd.read_csv(csv_path)

    # detectează automat coloana de predicție
    pred_col = [c for c in df.columns if "pred" in c.lower()][0]

    y_true = df["RUL"].values
    y_pred = df[pred_col].values

    abs_error = np.abs(y_true - y_pred)

    out_dir = Path("outputs/figures/ch6")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fig_6_11_error_distribution_{model_name.lower()}.png"

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(abs_error, bins=40)

    ax.set_xlabel("Eroare absolută (cicluri)")
    ax.set_ylabel("Frecvență")
    ax.set_title(f"Distribuția erorii absolute – {model_name}", pad=10)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()