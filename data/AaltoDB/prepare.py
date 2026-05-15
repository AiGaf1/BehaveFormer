import pickle
import random
from pathlib import Path

import pandas as pd

from data.AaltoDB import features
from utils.logger import get_logger

LOGGER = get_logger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "prep_data"

# Paths for source data and cached raw/feature pickle files.
PATH_CSV                        = _DATA_DIR / "keystroke_data.csv"
PATH_TRAINING_FEATURES_PICKLE   = _DATA_DIR / "training_features.pickle"
PATH_VALIDATION_FEATURES_PICKLE = _DATA_DIR / "validation_features.pickle"
PATH_TEST_FEATURES_PICKLE       = _DATA_DIR / "test_features.pickle"
PATH_RAW_TRAINING_PICKLE        = _DATA_DIR / "training_data.pickle"
PATH_RAW_VALIDATION_PICKLE      = _DATA_DIR / "validation_data.pickle"
PATH_RAW_TEST_PICKLE            = _DATA_DIR / "testing_data.pickle"

SESSIONS_PER_USER = 15
SPLIT_SEED        = 42


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load() -> tuple[list, list, list]:
    """Return (training_data, validation_data, test_data).
    Format: list[user][session] -> ndarray(n_keystrokes, 3): [hold_time, flight_time, key_id]
    """
    if (PATH_TRAINING_FEATURES_PICKLE.exists()
            and PATH_VALIDATION_FEATURES_PICKLE.exists()
            and PATH_TEST_FEATURES_PICKLE.exists()):
        LOGGER.info("Using cached Aalto feature pickles")
        return (
            _load_pickle(PATH_TRAINING_FEATURES_PICKLE),
            _load_pickle(PATH_VALIDATION_FEATURES_PICKLE),
            _load_pickle(PATH_TEST_FEATURES_PICKLE),
        )

    if (PATH_RAW_TRAINING_PICKLE.exists()
            and PATH_RAW_VALIDATION_PICKLE.exists()
            and PATH_RAW_TEST_PICKLE.exists()):
        LOGGER.info("Building Aalto feature pickles from cached raw pickles")
        training_data   = features.apply(_load_pickle(PATH_RAW_TRAINING_PICKLE))
        validation_data = features.apply(_load_pickle(PATH_RAW_VALIDATION_PICKLE))
        test_data       = features.apply(_load_pickle(PATH_RAW_TEST_PICKLE))
        _save_pickle(PATH_TRAINING_FEATURES_PICKLE, training_data)
        _save_pickle(PATH_VALIDATION_FEATURES_PICKLE, validation_data)
        _save_pickle(PATH_TEST_FEATURES_PICKLE, test_data)
        return training_data, validation_data, test_data

    LOGGER.info("Building Aalto feature pickles from CSV")
    data = pd.read_csv(PATH_CSV)
    data_dict = {
        user: [group[["press_time", "release_time", "key_code"]].to_numpy()
               for _, group in sessions.groupby("session_id")]
        for user, sessions in data.groupby("user_id")
    }
    all_users = [s for s in data_dict.values() if len(s) == SESSIONS_PER_USER]
    random.Random(SPLIT_SEED).shuffle(all_users)
    LOGGER.info("Users after filtering: %s", len(all_users))

    split_train = int(len(all_users) * 0.8)
    split_validation = int(len(all_users) * 0.9)

    training_data   = features.apply(all_users[:split_train])
    validation_data = features.apply(all_users[split_train:split_validation])
    test_data       = features.apply(all_users[split_validation:])
    _save_pickle(PATH_TRAINING_FEATURES_PICKLE, training_data)
    _save_pickle(PATH_VALIDATION_FEATURES_PICKLE, validation_data)
    _save_pickle(PATH_TEST_FEATURES_PICKLE, test_data)
    return training_data, validation_data, test_data
