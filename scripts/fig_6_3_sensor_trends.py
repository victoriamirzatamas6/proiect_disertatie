from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.data.cmapss_loader import load_cmapss


def infer_columns(df):
    """Detectează numele coloanelor standard pentru CMAPSS."""
    # coloana unității poate fi unit / unit_id / engine_id
    if "unit" in df.columns:
        unit_col = "unit"
    elif "unit_id" in df.columns:
        unit_col = "unit_id"
    elif "engine_id" in df.columns:
        unit_col = "engine_id"
    else:
        unit_col = None

    cycle_col = "cycle" if "cycle" in df.columns else None

    if unit_col is None or cycle_col is None:
        raise ValueError(
            f"Nu pot detecta coloanele unit/cycle. Coloane disponibile: {list(df.columns)}"
        )
    return unit_col, cycle_col


def pick_top_sensors_by_slope(unit_df, sensor_cols, k=3):
    """
    Alege k senzori cu |panta| cea mai mare (trend puternic) pentru unitatea aleasă.
    Panta este estimată cu regresie liniară simplă pe (cycle, sensor).
    """
    x = unit_df["cycle"].values.astype(float)
    x = x - x.mean()  # centrare pentru stabilitate numerică

    scores = []
    for s in sensor_cols:
        y = unit_df[s].values.astype(float)
        # dacă există NaN, ignoră senzorul (rar)
        if np.isnan(y).any():
            continue
        y = y - y.mean()
        denom = (x ** 2).sum()
        if denom == 0:
            continue
        slope = (x * y).sum() / denom
        scores.append((s, abs(slope)))

    scores.sort(key=lambda t: t[1], reverse=True)
    return [s for s, _ in scores[:k]]


def main():
    # ======= setări =======
    unit_id = 5      # schimbă cu 1..10 dacă vrei
    top_k = 3
    # ======================

    train_df, _, _ = load_cmapss(raw_dir="data/raw", fd="FD001")

    unit_col, cycle_col = infer_columns(train_df)

    # normalizează numele coloanei cycle la "cycle" ca să refolosim funcțiile
    if cycle_col != "cycle":
        train_df = train_df.rename(columns={cycle_col: "cycle"})
    if unit_col != "unit":
        train_df = train_df.rename(columns={unit_col: "unit"})

    # Senzori
    sensor_cols = [c for c in train_df.columns if c.startswith("s")]
    if not sensor_cols:
        raise ValueError("Nu am găsit coloane de senzori care încep cu 's'.")

    # elimină senzori constanți global
    nunique = train_df[sensor_cols].nunique(dropna=True)
    sensor_cols = [c for c in sensor_cols if nunique[c] > 1]

    # Selectează unitatea
    unit_df = train_df[train_df["unit"] == unit_id].sort_values("cycle")
    if unit_df.empty:
        raise ValueError(f"Unitatea {unit_id} nu există. Alege alt unit_id.")

    # Alege senzori cu trend puternic (panta mare)
    sensors = pick_top_sensors_by_slope(unit_df, sensor_cols, k=top_k)
    if len(sensors) < top_k:
        raise ValueError("Nu am putut selecta suficienți senzori. Încearcă alt unit_id.")

    # Plot (stil disertație)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    for s in sensors:
        ax.plot(unit_df["cycle"].values, unit_df[s].values, linewidth=1.8, label=s)

    ax.set_xlabel("Ciclu (cycle)")
    ax.set_ylabel("Valoare senzor")
    ax.set_title(
        f"Evoluția senzorilor pentru unitatea {unit_id} (CMAPSS FD001 – train)\n"
        f"Senzori selectați pe baza trendului (|panta| maximă): {', '.join(sensors)}",
        pad=10
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    out_dir = Path("outputs/figures/ch6")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fig_6_3_sensor_trends_unit{unit_id}.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()