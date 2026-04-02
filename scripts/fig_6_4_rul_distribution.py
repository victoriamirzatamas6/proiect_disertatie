from pathlib import Path
import matplotlib.pyplot as plt

from src.data.cmapss_loader import load_cmapss


def compute_rul_from_cycles(df, unit_col="unit_id", cycle_col="cycle"):
    """
    RUL = max_cycle(unit) - cycle
    Pentru train set, max_cycle e cunoscut din date.
    """
    max_cycle = df.groupby(unit_col)[cycle_col].max()
    rul = max_cycle.loc[df[unit_col]].values - df[cycle_col].values
    return rul


def main():
    # ===== setări =====
    fd = "FD001"
    rul_cap = 125   # pune None dacă NU vrei capare
    bins = 40
    # ==================

    train_df, _, _ = load_cmapss(raw_dir="data/raw", fd=fd)

    # La tine coloana e unit_id (am văzut din eroare)
    if "unit_id" not in train_df.columns or "cycle" not in train_df.columns:
        raise ValueError(f"Coloane așteptate lipsă. Coloane disponibile: {list(train_df.columns)}")

    rul = compute_rul_from_cycles(train_df, unit_col="unit_id", cycle_col="cycle")

    # Capare (folosită des în literatură pentru CMAPSS)
    if rul_cap is not None:
        rul = rul.clip(max=rul_cap)

    # Output
    out_dir = Path("outputs/figures/ch6")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_6_4_rul_distribution.png"

    # Stil disertație
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rul, bins=bins)
    ax.set_xlabel("RUL (cicluri)")
    ax.set_ylabel("Frecvență")
    title = f"Distribuția RUL – CMAPSS {fd} (train)"
    if rul_cap is not None:
        title += f" (capare la {rul_cap})"
    ax.set_title(title, pad=10)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()