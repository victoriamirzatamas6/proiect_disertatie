from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.data.cmapss_loader import load_cmapss


def main():
    # 1) Load train set (FD001) din data/raw
    train_df, _, _ = load_cmapss(raw_dir="data/raw", fd="FD001")

    # 2) Selectează doar coloanele de senzori
    sensor_cols = [c for c in train_df.columns if c.startswith("s")]
    X = train_df[sensor_cols]

    # Elimină senzori constanți (recomandat)
    nunique = X.nunique(dropna=True)
    sensor_cols = [c for c in sensor_cols if nunique[c] > 1]
    X = train_df[sensor_cols]

    # 3) Corelație Pearson
    corr = X.corr(method="pearson")

    # 4) Output
    out_dir = Path("outputs/figures/ch6")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_6_2_correlation.png"

    # 5) Stil pentru disertație
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 100,   # dpi de afișare; salvarea e la 300
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scală fixă [-1, 1] pentru interpretare corectă și comparabilitate
    im = ax.imshow(
        corr.values,
        vmin=-1, vmax=1,
        interpolation="nearest",
        aspect="equal",
        cmap="coolwarm"  # bun pentru corelații +/- (roșu/ albastru)
    )

    # Colorbar curat
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coeficient de corelație (Pearson)")

    # Tick-uri
    ax.set_xticks(np.arange(len(sensor_cols)))
    ax.set_yticks(np.arange(len(sensor_cols)))
    ax.set_xticklabels(sensor_cols, rotation=90)
    ax.set_yticklabels(sensor_cols)

    # (Opțional) linii fine pentru separare (ajută la print)
    ax.set_xticks(np.arange(-0.5, len(sensor_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sensor_cols), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Titlu academic
    ax.set_title("Matricea de corelație (Pearson) între senzorii CMAPSS FD001 (train)", pad=10)

    # Layout bun (fără spațiu inutil)
    fig.tight_layout()

    # Salvare 300 dpi
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()