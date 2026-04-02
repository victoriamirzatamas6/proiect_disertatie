import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PdM Platform+", layout="wide")
st.title("PdM Platform+ – CMAPSS FD001")

st.sidebar.header("Controls")

rul_xgb = pd.read_csv("outputs/predictions/rul_baseline_xgb.csv") if st.sidebar.checkbox("Load XGB", True) else None
rul_lstm = pd.read_csv("outputs/predictions/rul_lstm.csv") if st.sidebar.checkbox("Load LSTM", True) else None
anom = pd.read_csv("outputs/anomaly/anomaly_scores.csv")

with open("outputs/metrics/summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

units = sorted(set(anom["unit_id"].unique()))
unit = st.sidebar.selectbox("Select unit_id", units)

thr_mode = st.sidebar.selectbox("Anomaly threshold", ["AE p99", "PCA p99"])
if thr_mode == "AE p99":
    score_col, thr_col, flag_col = "anomaly_score_ae", "threshold_ae_p99", "is_anomaly_ae_p99"
else:
    score_col, thr_col, flag_col = "anomaly_score_pca", "threshold_pca_p99", "is_anomaly_pca_p99"

col1, col2 = st.columns(2)

with col1:
    st.subheader("RUL – XGBoost (default window=30)")
    if rul_xgb is not None:
        d = rul_xgb[rul_xgb["unit_id"] == unit].sort_values("cycle_end")
        fig = plt.figure()
        plt.plot(d["cycle_end"], d["RUL"], label="True")
        plt.plot(d["cycle_end"], d["pred_RUL_xgb"], label="Pred (XGB)")
        plt.xlabel("cycle"); plt.ylabel("RUL"); plt.legend()
        st.pyplot(fig)

with col2:
    st.subheader("RUL – LSTM (default window=30)")
    if rul_lstm is not None:
        d = rul_lstm[rul_lstm["unit_id"] == unit].sort_values("cycle_end")
        fig = plt.figure()
        plt.plot(d["cycle_end"], d["RUL"], label="True")
        plt.plot(d["cycle_end"], d["pred_RUL_lstm"], label="Pred (LSTM)")
        plt.xlabel("cycle"); plt.ylabel("RUL"); plt.legend()
        st.pyplot(fig)

st.subheader("Anomaly detection (scores + flags)")
a = anom[anom["unit_id"] == unit].sort_values("cycle")
thr = float(a[thr_col].iloc[0]) if len(a) else None

fig = plt.figure()
plt.plot(a["cycle"], a[score_col], label=score_col)
if thr is not None:
    plt.axhline(thr, linestyle="--", label="threshold")
plt.scatter(a.loc[a[flag_col] == 1, "cycle"], a.loc[a[flag_col] == 1, score_col], label="anomaly", s=12)
plt.xlabel("cycle"); plt.ylabel("score"); plt.legend()
st.pyplot(fig)

st.subheader("Top units by max anomaly score")
top = anom.groupby("unit_id")[score_col].max().sort_values(ascending=False).head(10).reset_index()
st.dataframe(top)

st.subheader("Downloads")
st.download_button("Download anomaly_scores.csv", anom.to_csv(index=False).encode("utf-8"), file_name="anomaly_scores.csv", mime="text/csv")
if rul_xgb is not None:
    st.download_button("Download rul_baseline_xgb.csv", rul_xgb.to_csv(index=False).encode("utf-8"), file_name="rul_baseline_xgb.csv", mime="text/csv")
if rul_lstm is not None:
    st.download_button("Download rul_lstm.csv", rul_lstm.to_csv(index=False).encode("utf-8"), file_name="rul_lstm.csv", mime="text/csv")

st.subheader("Summary")
st.json(summary)
