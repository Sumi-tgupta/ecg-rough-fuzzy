"""Run reproducible MIT-BIH experiments from the PulseLab repository root."""
from __future__ import annotations

import argparse
from pathlib import Path

import config
from modules import data_loader, pipeline


def _progress(fraction: float, message: str) -> None:
    print(f"[{fraction:6.1%}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate PulseLab's rough-fuzzy and baseline models on MIT-BIH records."
    )
    parser.add_argument("--download", action="store_true", help="download the selected MIT-BIH records from PhysioNet")
    parser.add_argument("--quick", action="store_true", help="use the first 8 DS1 and 6 DS2 records for a smaller engineering run")
    parser.add_argument("--no-baselines", action="store_true", help="skip Random Forest and RBF-SVM models")
    parser.add_argument("--no-save-models", action="store_true", help="save metrics only; do not serialize model artifacts")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="directory containing WFDB .hea/.dat/.atr records")
    parser.add_argument("--output-dir", type=Path, default=config.EXPERIMENTS_DIR, help="directory for experiment outputs")
    parser.add_argument("--run-name", default=None, help="optional unique experiment directory name")
    args = parser.parse_args()

    train_ids = config.DS1[:8] if args.quick else list(config.DS1)
    test_ids = config.DS2[:6] if args.quick else list(config.DS2)
    requested = train_ids + test_ids

    if args.download:
        print(f"Downloading {len(requested)} MIT-BIH records into {args.data_dir} ...")
        try:
            data_loader.download_records(requested, data_dir=args.data_dir, progress=_progress)
        except ImportError as exc:
            parser.error(f"WFDB is required for dataset downloads. Install requirements.txt first. Details: {exc}")

    available = set(data_loader.available_records(args.data_dir))
    missing = [rid for rid in requested if rid not in available]
    if missing:
        preview = ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "")
        print("Missing MIT-BIH records:", preview)
        print("Download the quick engineering subset with:")
        print("  py train.py --download --quick")
        print("Or download the complete DS1/DS2 experiment with:")
        print("  py train.py --download")
        return 2

    try:
        output = pipeline.run_experiment(
            train_ids,
            test_ids,
            data_dir=args.data_dir,
            output_root=args.output_dir,
            progress=_progress,
            include_baselines=not args.no_baselines,
            save_models=not args.no_save_models,
            run_name=args.run_name,
        )
    except Exception as exc:
        print(f"Experiment failed: {type(exc).__name__}: {exc}")
        return 1

    print("\nExperiment completed.")
    print("Artifacts:", output)
    print("Metrics:", output / "metrics.json")
    print("Comparison:", output / "model_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
