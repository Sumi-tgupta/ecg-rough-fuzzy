"""MIT-BIH Arrhythmia Database loading (WFDB) and AAMI label mapping."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import config
from config import AAMI_MAP, DATA_DIR, get_logger

log = get_logger("data_loader")


class RecordError(Exception):
    """Raised when a record is missing or invalid."""


@dataclass
class Record:
    record_id: str
    fs: float
    signal: np.ndarray          # (n_samples, n_channels) physical units (mV)
    channel_names: list
    ann_samples: np.ndarray     # sample index of each AAMI-mappable beat annotation
    ann_symbols: np.ndarray     # original MIT-BIH symbol
    ann_classes: np.ndarray     # AAMI class letter
    n_excluded: int             # non-beat / unmapped annotations skipped


def all_record_ids() -> list:
    return sorted(set(config.DS1 + config.DS2 + config.PACED_RECORDS), key=int)


def _record_files_complete(record_id: str, data_dir: Path) -> bool:
    base = Path(data_dir) / str(record_id)
    return base.with_suffix(".hea").exists() and base.with_suffix(".dat").exists() and base.with_suffix(".atr").exists()


def available_records(data_dir: Path = DATA_DIR) -> list:
    root = Path(data_dir)
    return sorted(
        {p.stem for p in root.glob("*.hea") if _record_files_complete(p.stem, root)},
        key=lambda s: int(s) if s.isdigit() else 0,
    )


def download_records(record_ids=None, data_dir: Path = DATA_DIR, progress=None) -> list:
    """Download records from PhysioNet; returns ids available locally afterwards."""
    import wfdb
    record_ids = list(record_ids or all_record_ids())
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    todo = [str(r) for r in record_ids if not _record_files_complete(str(r), Path(data_dir))]
    for i, rid in enumerate(todo):
        try:
            wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=[rid])
        except Exception as exc:
            log.error("download failed for %s: %s", rid, exc)
        if progress:
            progress((i + 1) / max(len(todo), 1), f"Downloading record {rid}")
    return [str(r) for r in record_ids if _record_files_complete(str(r), Path(data_dir))]


def load_record(record_id: str, data_dir: Path = DATA_DIR) -> Record:
    import wfdb
    rid = str(record_id)
    base = Path(data_dir) / rid
    if not _record_files_complete(rid, Path(data_dir)):
        raise RecordError(f"Record {rid} is missing one or more WFDB files (.hea, .dat, .atr) in {data_dir}. Download it first.")
    try:
        rec = wfdb.rdrecord(str(base))
        ann = wfdb.rdann(str(base), "atr")
    except Exception as exc:
        raise RecordError(f"Could not read record {rid}: {exc}") from exc
    if rec.p_signal is None or rec.p_signal.ndim != 2 or not np.isfinite(rec.p_signal).any():
        raise RecordError(f"Record {rid} has no valid signal")
    if not np.isfinite(rec.fs) or rec.fs <= 0:
        raise RecordError(f"Record {rid} has invalid sampling frequency: {rec.fs}")
    sig = np.nan_to_num(rec.p_signal.astype(np.float64), nan=0.0)
    samples, symbols = np.asarray(ann.sample), np.asarray(ann.symbol)
    keep = np.array([s in AAMI_MAP for s in symbols], dtype=bool)
    keep &= (samples >= 0) & (samples < sig.shape[0])
    return Record(rid, float(rec.fs), sig, list(rec.sig_name), samples[keep], symbols[keep],
                  np.array([AAMI_MAP[s] for s in symbols[keep]]), int((~keep).sum()))


def annotation_summary(record_ids, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for rid in record_ids:
        try:
            r = load_record(rid, data_dir)
        except RecordError as exc:
            log.warning(str(exc)); continue
        row = {"record": rid, "fs": r.fs, "channels": "/".join(r.channel_names),
               "n_beats": len(r.ann_samples), "excluded_annotations": r.n_excluded}
        for c in config.CLASSES:
            row[c] = int((r.ann_classes == c).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def excluded_symbol_table() -> pd.DataFrame:
    return pd.DataFrame([{"symbol": k, "meaning": v} for k, v in config.EXCLUDED_SYMBOLS.items()])


def aami_table() -> pd.DataFrame:
    return pd.DataFrame([{"symbol": k, "AAMI class": v} for k, v in config.AAMI_MAP.items()])
