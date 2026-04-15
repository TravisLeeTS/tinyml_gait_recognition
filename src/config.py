from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

UCI_HAR_DIR = RAW_DATA_DIR / "UCI HAR Dataset"
UCI_HAR_ZIP = RAW_DATA_DIR / "UCI_HAR_Dataset.zip"
UCI_HAR_URLS = [
    # Primary official UCI endpoint. This returned HTTP 502 during package creation.
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    # Public course mirror used only when UCI is temporarily unavailable.
    "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip",
]

WISDM_CLASSIC_TGZ = RAW_DATA_DIR / "WISDM_ar_latest.tar.gz"
WISDM_CLASSIC_DIR = RAW_DATA_DIR / "WISDM_ar_v1.1"
WISDM_CLASSIC_URL = (
    "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
)

UCI_HAR_CLASSES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

WISDM_CLASSIC_CLASSES = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing",
]

UCI_HAR_SEQUENCE_SIGNALS = [
    # Total acceleration preserves gravity/posture cues needed for SITTING,
    # STANDING, and LAYING and is closer to Arduino IMU collection.
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

UCI_HAR_TIMESTEPS = 128
UCI_HAR_CHANNELS = len(UCI_HAR_SEQUENCE_SIGNALS)
UCI_HAR_SAMPLING_HZ = 50
UCI_HAR_WINDOW_SECONDS = 2.56
UCI_HAR_OVERLAP = 0.50

WISDM_CLASSIC_SAMPLING_HZ = 20
WISDM_CLASSIC_WINDOW_SECONDS = 4.0
WISDM_CLASSIC_OVERLAP = 0.50

DEFAULT_DEVICE = "cpu"
