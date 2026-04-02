import argparse
from src.experiments.report import build_report, html_to_pdf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--config_path", default="configs/config.yaml")
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--pdf", action="store_true", help="Also export PDF (requires weasyprint)")
    args = ap.parse_args()

    html_path = build_report(outputs_dir=args.outputs_dir, config_path=args.config_path, fd=args.fd)
    print("Report HTML:", html_path)
    if args.pdf:
        pdf_path = html_to_pdf(html_path)
        print("Report PDF :", pdf_path)

if __name__ == "__main__":
    main()
