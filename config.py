"""Central configuration for the Rough-Fuzzy ECG project."""
from __future__ import annotations
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mitdb"
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments"
for _d in (DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
FS = 360                      # MIT-BIH sampling frequency (Hz)
BANDPASS = (0.5, 40.0)        # Hz
BASELINE_MEDIAN_WINDOWS = (0.2, 0.6)  # seconds (two-stage median filter)
NOTCH_HZ = 50.0               # set 60.0 for 60 Hz mains; applied only if enabled
PRE_R_SEC = 0.25              # beat window before R peak
POST_R_SEC = 0.40             # beat window after R peak
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_NAMES = {
    "N": "Normal / bundle-branch-block beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown / paced beat",
}

# ---- AAMI EC57 mapping of MIT-BIH annotation symbols -------------------------
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
# Symbols that are NOT heartbeats (rhythm / signal-quality / comment markers) -> excluded
EXCLUDED_SYMBOLS = {
    "[": "start of ventricular flutter/fibrillation", "!": "ventricular flutter wave",
    "]": "end of ventricular flutter/fibrillation", "x": "non-conducted P-wave (blocked APC)",
    "(": "waveform onset", ")": "waveform end", "p": "peak of P-wave", "t": "peak of T-wave",
    "u": "peak of U-wave", "`": "PQ junction", "'": "J-point", "^": "non-captured pacemaker artifact",
    "|": "isolated QRS-like artifact", "~": "signal quality change", "+": "rhythm change",
    "s": "systole", "T": "T-wave change", "*": "systole", "D": "diastole", "=": "J-point",
    '"': "comment annotation", "@": "link to external data", "?": "unclassifiable (not a beat label in AAMI)",
}
# Records containing paced rhythm are excluded by AAMI recommendation
PACED_RECORDS = ["102", "104", "107", "217"]
# Inter-patient split of de Chazal et al. (2004): DS1 = train, DS2 = test
DS1 = ["101","106","108","109","112","114","115","116","118","119","122","124",
       "201","203","205","207","208","209","215","220","223","230"]
DS2 = ["100","103","105","111","113","117","121","123","200","202","210","212",
       "213","214","219","221","222","228","231","232","233","234"]

# ---- model hyper-parameters ---------------------------------------------------
ROUGH_BINS = 4                # equal-frequency discretisation bins
ROUGH_MAX_PER_CLASS = 2500    # class-balanced sample used for reduct search (train only)
ROUGH_TOLERANCE = 0.0         # stop when gamma(R) >= gamma(C) - tolerance
FUZZY_MAX_PER_CLASS = 3000
FUZZY_RULES_PER_CLASS = 12
FUZZY_MAX_ANTECEDENTS = 3
FUZZY_BEAM = 30
FUZZY_MIN_SUPPORT = 0.03
MIN_TRAIN_SAMPLES_PER_CLASS = 20

DISCLAIMER = ("Academic research prototype. Outputs are NOT medically validated and must NOT be used "
              "for clinical diagnosis or treatment decisions.")

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)
