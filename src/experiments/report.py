from __future__ import annotations
from pathlib import Path
import base64
import pandas as pd
import json
import datetime

def _img_to_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)

def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_html_report(out_dir: str = "outputs") -> str:
    out = Path(out_dir)
    report_dir = out / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = _safe_read_json(out / "metrics" / "summary.json") or {}
    ablation = _safe_read_csv(out / "metrics" / "ablation_results.csv")
    alarm = _safe_read_csv(out / "metrics" / "anomaly_alarm_rates.csv")

    imgs = {
        "Error histogram — XGB": _img_to_data_uri(out / "figures" / "error_hist_xgb.png"),
        "Error histogram — LSTM": _img_to_data_uri(out / "figures" / "error_hist_lstm.png"),
        "XGB gain importance": _img_to_data_uri(out / "figures" / "xgb_gain_importance.png"),
        "XGB permutation importance": _img_to_data_uri(out / "figures" / "xgb_permutation_importance.png"),
        "AE score hist (train vs test)": _img_to_data_uri(out / "figures" / "ae_score_hist.png"),
        "PCA score hist (train vs test)": _img_to_data_uri(out / "figures" / "pca_score_hist.png"),
    }

    css = """
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    h1,h2,h3 { margin-top: 28px; }
    .meta { color: #444; font-size: 0.95em; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0 18px 0; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 0.95em; }
    th { background: #f6f6f6; text-align: left; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .img { width: 100%; border: 1px solid #ddd; border-radius: 8px; }
    .small { font-size: 0.92em; color: #333; }
    pre { margin: 0; }
    """

    def df_to_html(df: pd.DataFrame | None, max_rows: int = 60) -> str:
        if df is None:
            return "<p class='small'>Fișier indisponibil (încă). Rulează pipeline-ul întâi.</p>"
        d = df.copy()
        if len(d) > max_rows:
            d = d.head(max_rows)
        return d.to_html(index=False, escape=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append(f"<style>{css}</style></head><body>")
    html.append("<h1>Raport experimental – PdM Platform+ (CMAPSS FD001)</h1>")
    html.append(f"<div class='meta'>Generat la: {ts}</div>")

    html.append("<h2>1. Rezumat rulare</h2>")
    html.append("<div class='card'><pre>")
    html.append(json.dumps(summary, indent=2, ensure_ascii=False))
    html.append("</pre></div>")

    html.append("<h2>2. Ablation – window size</h2>")
    html.append("<div class='card'>")
    html.append(df_to_html(ablation))
    html.append("</div>")

    html.append("<h2>3. Interpretabilitate – XGBoost</h2>")
    html.append("<div class='grid'>")
    for title in ["XGB gain importance", "XGB permutation importance"]:
        uri = imgs.get(title)
        if uri:
            html.append(f"<div class='card'><h3>{title}</h3><img class='img' src='{uri}'/></div>")
        else:
            html.append(f"<div class='card'><h3>{title}</h3><p class='small'>Imagine indisponibilă.</p></div>")
    html.append("</div>")

    html.append("<h2>4. RUL – distribuția erorilor</h2>")
    html.append("<div class='grid'>")
    for title in ["Error histogram — XGB", "Error histogram — LSTM"]:
        uri = imgs.get(title)
        if uri:
            html.append(f"<div class='card'><h3>{title}</h3><img class='img' src='{uri}'/></div>")
        else:
            html.append(f"<div class='card'><h3>{title}</h3><p class='small'>Imagine indisponibilă.</p></div>")
    html.append("</div>")

    html.append("<h2>5. Anomaly detection – AE vs PCA</h2>")
    html.append("<h3>5.1 Praguri și alarm-rate</h3>")
    html.append("<div class='card'>")
    html.append(df_to_html(alarm))
    html.append("</div>")

    html.append("<h3>5.2 Distribuția scorurilor</h3>")
    html.append("<div class='grid'>")
    for title in ["AE score hist (train vs test)", "PCA score hist (train vs test)"]:
        uri = imgs.get(title)
        if uri:
            html.append(f"<div class='card'><h3>{title}</h3><img class='img' src='{uri}'/></div>")
        else:
            html.append(f"<div class='card'><h3>{title}</h3><p class='small'>Imagine indisponibilă.</p></div>")
    html.append("</div>")

    html.append("<hr/><div class='small'>Notă: raportul include doar fișierele existente în outputs/. Dacă unele lipsesc, rulează pipeline-ul și regenerează raportul.</div>")
    html.append("</body></html>")

    report_path = report_dir / "report.html"
    report_path.write_text("\n".join(html), encoding="utf-8")
    return str(report_path)

def generate_pdf_report(out_dir: str = "outputs") -> str | None:
    # Optional PDF using reportlab. If not installed, return None.
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception:
        return None

    out = Path(out_dir)
    report_dir = out / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / "report.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    y = height - 40

    def draw_title(txt: str, size: int = 16):
        nonlocal y
        c.setFont("Helvetica-Bold", size)
        c.drawString(40, y, txt)
        y -= size + 10

    def draw_text(txt: str, size: int = 10):
        nonlocal y
        c.setFont("Helvetica", size)
        c.drawString(40, y, txt[:120])
        y -= size + 6

    draw_title("Raport experimental – PdM Platform+ (CMAPSS FD001)", 16)
    draw_text(f"Generat la: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 10)
    y -= 10

    figs = [
        ("XGB gain importance", out/"figures"/"xgb_gain_importance.png"),
        ("XGB permutation importance", out/"figures"/"xgb_permutation_importance.png"),
        ("AE score hist", out/"figures"/"ae_score_hist.png"),
        ("PCA score hist", out/"figures"/"pca_score_hist.png"),
    ]

    for name, p in figs:
        if y < 220:
            c.showPage()
            y = height - 40
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, name)
        y -= 14
        if p.exists():
            img = ImageReader(str(p))
            iw, ih = img.getSize()
            max_w = width - 80
            max_h = 180
            scale = min(max_w/iw, max_h/ih)
            w = iw * scale
            h = ih * scale
            c.drawImage(img, 40, y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
            y -= h + 18
        else:
            draw_text("Imagine indisponibilă.", 10)

    c.save()
    return str(pdf_path)


# ---- Compatibility wrappers used by the pipeline ----
def build_report(outputs_dir: str = "outputs", config_path: str | None = None, fd: str | None = None) -> str:
    # config_path and fd are accepted for traceability; HTML report is built from files in outputs_dir.
    return generate_html_report(outputs_dir)

def html_to_pdf(html_path: str) -> str | None:
    # Convert to PDF by regenerating a lightweight PDF from outputs/figures.
    # Derive outputs_dir from .../outputs/report/report.html
    try:
        out_dir = str(Path(html_path).resolve().parents[1])
    except Exception:
        out_dir = "outputs"
    return generate_pdf_report(out_dir)
