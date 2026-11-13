"""CLI: python train.py [--quick] [--download] -> runs the full experiment on real MIT-BIH records."""
import argparse
import config
from modules import data_loader as dl, pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate Rough-Fuzzy ECG models on MIT-BIH.")
    ap.add_argument("--download", action="store_true", help="download missing records from PhysioNet first")
    ap.add_argument("--quick", action="store_true", help="use a small record subset (8 train / 6 test) for a fast run")
    args = ap.parse_args()
    tr, te = config.DS1, config.DS2
    if args.quick:
        tr, te = tr[:8], te[:6]
    if args.download:
        dl.download_records(tr + te, progress=lambda p, m: print(f"{p:5.0%} {m}"))
    out = pipeline.run_experiment(tr, te, progress=lambda p, m: print(f"{p:5.0%} {m}"))
    print("Experiment saved to", out)
