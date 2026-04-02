import argparse
import shutil
import joblib
import torch
import pandas as pd
import numpy as np

from src.utils.io import load_config, ensure_dirs, save_json
from src.utils.seed import set_seed
from src.utils.metrics import mae, rmse
from src.utils.plotting import save_pred_plot, save_error_hist, save_score_hist, save_feature_importance

from src.data.cmapss_loader import load_cmapss
from src.data.split import split_by_units
from src.data.preprocess import drop_constant_columns, fit_scaler, apply_scaler
from src.data.features import make_tabular_window_features, last_window_per_unit

from src.models.baseline_xgb import train_xgb, predict
from src.models.explain_xgb import xgb_gain_importance, permutation_importance_reg
from src.models.lstm_rul import make_sequences, train_lstm, predict_lstm
from src.models.anomaly_autoencoder import make_flat_windows as ae_make_flat_windows, fit_and_score as ae_fit_and_score
from src.models.anomaly_pca import fit_pca_reconstruction, reconstruction_error as pca_recon_err

from src.experiments.tracking import new_run_dir, snapshot_config, write_run_summary, append_runs_csv

def add_rul_train(df: pd.DataFrame, cap: int | None):
    last = df.groupby("unit_id")["cycle"].max().rename("last_cycle")
    out = df.merge(last, on="unit_id", how="left")
    out["RUL"] = (out["last_cycle"] - out["cycle"]).astype(int)
    out.drop(columns=["last_cycle"], inplace=True)
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=int(cap))
    return out

def add_rul_test(df_test: pd.DataFrame, rul_test: pd.DataFrame, cap: int | None):
    rul_map = {i + 1: int(rul_test.iloc[i, 0]) for i in range(len(rul_test))}
    last = df_test.groupby("unit_id")["cycle"].max().rename("last_cycle")
    out = df_test.merge(last, on="unit_id", how="left")
    out["RUL_last"] = out["unit_id"].map(rul_map).astype(int)
    out["RUL"] = (out["RUL_last"] + (out["last_cycle"] - out["cycle"])).astype(int)
    out.drop(columns=["last_cycle", "RUL_last"], inplace=True)
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=int(cap))
    return out

def unit_level_metrics(df_windows: pd.DataFrame, pred_col: str):
    last_df = last_window_per_unit(df_windows)
    y = last_df["RUL"].values
    p = last_df[pred_col].values
    return {"mae_unit_last": mae(y, p), "rmse_unit_last": rmse(y, p), "n_units": int(last_df.shape[0])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    ensure_dirs("outputs/models","outputs/metrics","outputs/predictions","outputs/anomaly","outputs/figures","outputs/runs")

    run_dir = None
    if cfg.get("tracking", {}).get("enabled", True):
        run_dir = new_run_dir()
        snapshot_config(args.config, run_dir)

    train_raw, test_raw, rul_test = load_cmapss(cfg["data"]["raw_dir"], cfg["data"]["fd"])
    cap = cfg["rul"]["cap"]
    cap = None if cap is None else int(cap)

    train_raw = add_rul_train(train_raw, cap)
    test_raw  = add_rul_test(test_raw, rul_test, cap)

    df_train, df_valid, train_units, valid_units = split_by_units(train_raw, float(cfg["split"]["train_units_ratio"]), int(cfg["seed"]))

    sensor_cols = [c for c in train_raw.columns if c.startswith("s")]
    feat_cols = ["op1","op2","op3"] + sensor_cols
    if bool(cfg["preprocess"]["drop_constant_cols"]):
        feat_cols = drop_constant_columns(df_train, feat_cols)

    scaler = fit_scaler(df_train, feat_cols, cfg["preprocess"]["scaler"])
    df_train_s = apply_scaler(df_train, feat_cols, scaler)
    df_valid_s = apply_scaler(df_valid, feat_cols, scaler)
    df_test_s  = apply_scaler(test_raw,  feat_cols, scaler)
    joblib.dump({"scaler": scaler, "feature_cols": feat_cols}, "outputs/models/preprocess.joblib")

    # Ablations
    step = int(cfg["features"]["step"])
    rows = []

    # XGB
    for w in cfg["features"]["window_sizes"]:
        tr = make_tabular_window_features(df_train_s, feat_cols, int(w), step, cap)
        va = make_tabular_window_features(df_valid_s, feat_cols, int(w), step, cap)
        te = make_tabular_window_features(df_test_s,  feat_cols, int(w), step, cap)

        feature_names = [c for c in tr.columns if c not in ["unit_id","cycle_end","RUL"]]
        Xtr, ytr = tr[feature_names].values, tr["RUL"].values
        Xva, yva = va[feature_names].values, va["RUL"].values
        Xte, yte = te[feature_names].values, te["RUL"].values

        model = train_xgb(Xtr, ytr, cfg["baseline"]["params"], seed=int(cfg["seed"]))
        pred = predict(model, Xte)

        out = te[["unit_id","cycle_end","RUL"]].copy()
        out["pred_RUL_xgb"] = pred

        m = {"model":"xgb","window":int(w),"mae":mae(yte,pred),"rmse":rmse(yte,pred)}
        m.update(unit_level_metrics(out, "pred_RUL_xgb"))
        rows.append(m)

        if int(w) == 30:
            model.save_model("outputs/models/xgb.json")
            out.to_csv("outputs/predictions/rul_baseline_xgb.csv", index=False)
            save_error_hist(yte, pred, "Error histogram — XGB", "outputs/figures/error_hist_xgb.png")
            # plots
            for uid in sorted(out["unit_id"].unique())[:5]:
                d = out[out["unit_id"] == uid]
                save_pred_plot(d, "cycle_end", "RUL", "pred_RUL_xgb", f"RUL XGB (unit {uid})", f"outputs/figures/rul_xgb_unit{uid}.png")
            # explainability
            gain = xgb_gain_importance(model)
            save_feature_importance(feature_names, gain, "XGB feature importance (gain)", "outputs/figures/xgb_gain_importance.png", top_n=15)
            n_sub = min(5000, len(Xva))
            perm = permutation_importance_reg(model, Xva[:n_sub], yva[:n_sub], seed=int(cfg["seed"]), n_repeats=5)
            save_feature_importance(feature_names, perm, "XGB permutation importance", "outputs/figures/xgb_permutation_importance.png", top_n=15)

    # LSTM
    for w in cfg["lstm"]["window_sizes"]:
        lstm_cfg = dict(cfg["lstm"])
        lstm_cfg["window_size"] = int(w)

        Xtr, ytr, _ = make_sequences(df_train_s, feat_cols, int(w), cap)
        Xva, yva, _ = make_sequences(df_valid_s, feat_cols, int(w), cap)
        Xte, yte, meta = make_sequences(df_test_s,  feat_cols, int(w), cap)

        model = train_lstm(Xtr, ytr, Xva, yva, lstm_cfg)
        pred = predict_lstm(model, Xte)

        out = pd.DataFrame(meta, columns=["unit_id","cycle_end"])
        out["RUL"] = yte
        out["pred_RUL_lstm"] = pred

        m = {"model":"lstm","window":int(w),"mae":mae(yte,pred),"rmse":rmse(yte,pred)}
        m.update(unit_level_metrics(out, "pred_RUL_lstm"))
        rows.append(m)

        if int(w) == 30:
            torch.save(model.state_dict(), "outputs/models/lstm.pt")
            out.to_csv("outputs/predictions/rul_lstm.csv", index=False)
            save_error_hist(yte, pred, "Error histogram — LSTM", "outputs/figures/error_hist_lstm.png")
            for uid in sorted(out["unit_id"].unique())[:5]:
                d = out[out["unit_id"] == uid]
                save_pred_plot(d, "cycle_end", "RUL", "pred_RUL_lstm", f"RUL LSTM (unit {uid})", f"outputs/figures/rul_lstm_unit{uid}.png")

    pd.DataFrame(rows).to_csv("outputs/metrics/ablation_results.csv", index=False)

    # Anomalies: AE + PCA
    an_cfg = cfg["anomaly"]
    ae_w = int(an_cfg["window_size"])
    Xtr_flat, _ = ae_make_flat_windows(df_train_s, feat_cols, ae_w)
    Xte_flat, meta_te = ae_make_flat_windows(df_test_s,  feat_cols, ae_w)

    ae_model, ae_train, ae_test, meta_te = ae_fit_and_score(df_train_s, df_test_s, feat_cols, an_cfg)
    torch.save(ae_model.state_dict(), "outputs/models/ae.pt")
    save_score_hist(ae_train, ae_test, "AE score distribution (train vs test)", "outputs/figures/ae_score_hist.png")

    pca = fit_pca_reconstruction(Xtr_flat, n_components=10, seed=int(cfg["seed"]))
    pca_train = pca_recon_err(pca, Xtr_flat)
    pca_test  = pca_recon_err(pca, Xte_flat)
    save_score_hist(pca_train, pca_test, "PCA score distribution (train vs test)", "outputs/figures/pca_score_hist.png")

    records = []
    for p in an_cfg["threshold_percentiles"]:
        thr_ae = float(np.percentile(ae_train, float(p)))
        thr_pca = float(np.percentile(pca_train, float(p)))
        records.append({"method":"AE","percentile":int(p),"threshold":thr_ae,"alarm_rate_test":float(np.mean(ae_test >= thr_ae))})
        records.append({"method":"PCA","percentile":int(p),"threshold":thr_pca,"alarm_rate_test":float(np.mean(pca_test >= thr_pca))})
    pd.DataFrame(records).to_csv("outputs/metrics/anomaly_alarm_rates.csv", index=False)

    thr_ae_99 = float(np.percentile(ae_train, 99.0))
    thr_pca_99 = float(np.percentile(pca_train, 99.0))
    an = pd.DataFrame(meta_te, columns=["unit_id","cycle"])
    an["anomaly_score_ae"] = ae_test
    an["threshold_ae_p99"] = thr_ae_99
    an["is_anomaly_ae_p99"] = (an["anomaly_score_ae"] >= thr_ae_99).astype(int)
    an["anomaly_score_pca"] = pca_test
    an["threshold_pca_p99"] = thr_pca_99
    an["is_anomaly_pca_p99"] = (an["anomaly_score_pca"] >= thr_pca_99).astype(int)
    an.to_csv("outputs/anomaly/anomaly_scores.csv", index=False)

    summary = {
        "cap_rul": cap,
        "n_features": len(feat_cols),
        "train_units": len(train_units),
        "valid_units": len(valid_units),
        "ablation_results_csv": "outputs/metrics/ablation_results.csv",
        "anomaly_alarm_rates_csv": "outputs/metrics/anomaly_alarm_rates.csv",
    }
    save_json("outputs/metrics/summary.json", summary)

    if run_dir is not None:
        for f in [
            "outputs/metrics/ablation_results.csv",
            "outputs/metrics/anomaly_alarm_rates.csv",
            "outputs/metrics/summary.json",
            "outputs/figures/xgb_gain_importance.png",
            "outputs/figures/xgb_permutation_importance.png",
            "outputs/figures/ae_score_hist.png",
            "outputs/figures/pca_score_hist.png",
        ]:
            try:
                shutil.copy2(f, f"{run_dir}/{Path(f).name}")
            except Exception:
                pass
        run_summary = {"run_dir": run_dir, "n_features": len(feat_cols), "cap_rul": cap, "train_units": len(train_units), "valid_units": len(valid_units)}
        write_run_summary(run_dir, run_summary)
        append_runs_csv(run_summary)

    
    # Generate report (HTML + try PDF)
    try:
        from src.experiments.report import build_report, html_to_pdf
        html_path = build_report(outputs_dir="outputs", config_path=args.config, fd=cfg["data"]["fd"])
        try:
            pdf_path = html_to_pdf(html_path)
        except Exception:
            pdf_path = None
        if run_dir is not None:
            try:
                shutil.copy2(html_path, f"{run_dir}/report.html")
                if pdf_path:
                    shutil.copy2(pdf_path, f"{run_dir}/report.pdf")
            except Exception:
                pass
    except Exception:
        pass

    print("DONE. See outputs/")

if __name__ == "__main__":
    main()
