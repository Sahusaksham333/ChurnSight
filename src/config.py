"""Project-wide constants. Anything you might want to tweak lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TARGET = "churn"

# The two exports in data/ name the identifier differently.
ID_COLUMNS = ("userid", "user")

# Unknown categories are stored as the literal string "na" in the raw files.
UNKNOWN_LABEL = "na"

RANDOM_STATE = 0
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_FEATURES = 20
DEFAULT_CV_FOLDS = 5

# Columns missing more than this share of values are dropped outright.
MAX_MISSING_SHARE = 0.10
MIN_VALID_CREDIT_SCORE = 300

# Probability cut-offs used to label customers in the export table.
RISK_BANDS = (0.33, 0.66)
