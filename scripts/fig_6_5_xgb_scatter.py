from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # === cale către predicții ===
    csv_path = Path("outputs/predictions/rul_baseline_xgb.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"Nu găsesc fișierul: {csv_path}")

    df = pd.read_csv(csv_path)

    # Verificare coloane
    if "RUL" not in df.columns or "pred_RUL_xgb" not in df.columns:
        raise ValueError(f"Coloane disponibile: {list(df.columns)}")

    y_true = df["RUL"].values
    y_pred = df["pred_RUL_xgb"].values

    # Output folder
    out_dir = Path("outputs/figures/ch6")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_6_5_xgb_pred_vs_true.png"

    # Stil academic
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Scatter
    ax.scatter(y_true, y_pred, alpha=0.5)

    # Linie ideală y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)

    ax.set_xlabel("RUL real (cicluri)")
    ax.set_ylabel("RUL estimat (cicluri)")
    ax.set_title("Predicție vs. RUL real – Model XGBoost (set test)", pad=10)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()