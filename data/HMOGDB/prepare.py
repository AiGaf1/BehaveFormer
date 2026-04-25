import pickle
from pathlib import Path

from data.HMOGDB.build import build_pickles, maybe_download
from experiments.common.logger import get_logger

LOGGER = get_logger(__name__)

_PREP_DIR        = Path(__file__).resolve().parent / "prep_data"
_TRAINING_FILE   = "training_keystroke_imu_data_all.pickle"
_VALIDATION_FILE = "validation_keystroke_imu_data_all.pickle"
_TESTING_FILE    = "testing_keystroke_imu_data_all.pickle"
_VOCAB_FILE      = "key_vocab.pickle"

SESSIONS_PER_USER = 8
ENROLL_SESSIONS   = 3
VERIFY_SESSIONS   = 5


def load(preprocessed_cfg: dict, validation_sequences_per_session: int,
         keystroke_sequence_len: int = 50, imu_sequence_len: int = 100,
         windowing_offset: int = 5) -> tuple[list, list, int]:
    """Return (training_data, validation_data, vocab_size).
    Format: list[user][session][sequence] -> [keystroke_ndarray, imu_ndarray]

    Validation sessions are trimmed to `validation_sequences_per_session` sequences each.
    """
    training_path   = _PREP_DIR / _TRAINING_FILE
    validation_path = _PREP_DIR / _VALIDATION_FILE

    maybe_download(preprocessed_cfg.get("train", ""), training_path)
    maybe_download(preprocessed_cfg.get("val",   ""), validation_path)
    maybe_download(preprocessed_cfg.get("test",  ""), _PREP_DIR / _TESTING_FILE)

    if not training_path.exists() or not validation_path.exists():
        LOGGER.info("HMOG pickles not found — building from raw dataset")
        build_pickles(keystroke_sequence_len, imu_sequence_len, windowing_offset)

    with open(training_path, "rb") as f:
        training_data = pickle.load(f)
    with open(validation_path, "rb") as f:
        validation_data = pickle.load(f)

    for user in validation_data:
        for idx, session in enumerate(user):
            user[idx] = session[:validation_sequences_per_session]

    with open(_PREP_DIR / _VOCAB_FILE, "rb") as f:
        key_vocab = pickle.load(f)
    vocab_size = len(key_vocab) + 1  # +1 for padding index 0

    LOGGER.info("HMOG loaded: %s train users, %s val users, vocab_size=%s",
                len(training_data), len(validation_data), vocab_size)
    return training_data, validation_data, vocab_size
