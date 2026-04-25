import pickle
from pathlib import Path

import pandas as pd

from data.AaltoDB import features
from experiments.common.logger import get_logger

LOGGER = get_logger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "prep_data"

_CSV                        = _DATA_DIR / "keystroke_data.csv"
_TRAINING_FEATURES_PICKLE   = _DATA_DIR / "training_features.pickle"
_VALIDATION_FEATURES_PICKLE = _DATA_DIR / "validation_features.pickle"
_RAW_TRAINING_PICKLE        = _DATA_DIR / "training_data.pickle"
_RAW_VALIDATION_PICKLE      = _DATA_DIR / "validation_data.pickle"

SESSIONS_PER_USER = 15
ENROLL_SESSIONS   = 10
VERIFY_SESSIONS   = 5


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load() -> tuple[list, list]:
    """Return (training_data, validation_data).
    Format: list[user][session] -> ndarray(n_keystrokes, 3): [hold_time, flight_time, key_id]
    """
    if _TRAINING_FEATURES_PICKLE.exists() and _VALIDATION_FEATURES_PICKLE.exists():
        LOGGER.info("Using cached Aalto feature pickles")
        return _load_pickle(_TRAINING_FEATURES_PICKLE), _load_pickle(_VALIDATION_FEATURES_PICKLE)

    if _RAW_TRAINING_PICKLE.exists() and _RAW_VALIDATION_PICKLE.exists():
        LOGGER.info("Building Aalto feature pickles from cached raw pickles")
        training_data   = features.apply(_load_pickle(_RAW_TRAINING_PICKLE))
        validation_data = features.apply(_load_pickle(_RAW_VALIDATION_PICKLE))
        _save_pickle(_TRAINING_FEATURES_PICKLE, training_data)
        _save_pickle(_VALIDATION_FEATURES_PICKLE, validation_data)
        return training_data, validation_data

    LOGGER.info("Building Aalto feature pickles from CSV")
    data = pd.read_csv(_CSV)
    data_dict = {
        user: [group[["press_time", "release_time", "key_code"]].to_numpy()
               for _, group in sessions.groupby("session_id")]
        for user, sessions in data.groupby("user_id")
    }
    all_users = [s for s in data_dict.values() if len(s) == SESSIONS_PER_USER]
    LOGGER.info("Users after filtering: %s", len(all_users))

    training_data   = features.apply(all_users[:-1050])
    validation_data = features.apply(all_users[-1050:-1000])
    _save_pickle(_TRAINING_FEATURES_PICKLE, training_data)
    _save_pickle(_VALIDATION_FEATURES_PICKLE, validation_data)
    return training_data, validation_data
