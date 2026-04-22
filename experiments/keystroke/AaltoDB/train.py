import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Config import Config
from metrics import Metric
from model import KeystrokeTransformer
from torch.utils.data import Dataset

from experiments.keystroke.common.lightning import run_keystroke_training

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

for path in ["utils", "evaluation", ""]:
    sys.path.append(str(PROJECT_ROOT / path))


DATA_DIR = PROJECT_ROOT / "data" / "AaltoDB" / "prep_data"
BEST_MODELS_DIR = HERE / "best_models"

CSV_PATH = DATA_DIR / "keystroke_data.csv"
TRAINING_FEATURES_PICKLE = DATA_DIR / "training_features.pickle"
VALIDATION_FEATURES_PICKLE = DATA_DIR / "validation_features.pickle"

SESSIONS_PER_USER = 15
ENROLL_SESSIONS = 10
VERIFY_SESSIONS = 5

for path in ["utils", "evaluation", ""]:
    sys.path.append(str(PROJECT_ROOT / path))


def _pad_sequence(sequence, seq_len):
    out = np.zeros((seq_len, sequence.shape[1]))
    out[:len(sequence)] = sequence[:seq_len]
    return out


def _compute_features(sequence):
    length = len(sequence)
    features = np.zeros((length, 10))
    for index, (press, release, key) in enumerate(sequence):
        next_item = sequence[index + 1] if index < length - 1 else None
        next_two = sequence[index + 2] if index < length - 2 else None
        features[index] = [
            (release - press) / 1000,
            (next_item[0] - release) / 1000 if next_item is not None else 0.0,
            (next_item[0] - press) / 1000 if next_item is not None else 0.0,
            (next_item[1] - release) / 1000 if next_item is not None else 0.0,
            (next_item[1] - press) / 1000 if next_item is not None else 0.0,
            (next_two[0] - release) / 1000 if next_two is not None else 0.0,
            (next_two[0] - press) / 1000 if next_two is not None else 0.0,
            (next_two[1] - release) / 1000 if next_two is not None else 0.0,
            (next_two[1] - press) / 1000 if next_two is not None else 0.0,
            key / 255,
        ]
    return features


def _load_data():

    def _load_pickle(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    
    if TRAINING_FEATURES_PICKLE.exists() and VALIDATION_FEATURES_PICKLE.exists():
        print("Using cached Aalto feature pickles")
        return _load_pickle(TRAINING_FEATURES_PICKLE), _load_pickle(VALIDATION_FEATURES_PICKLE)

    print("Building Aalto feature pickles")
    data = pd.read_csv(CSV_PATH)
    assert not data.isnull().values.any()
    data_dict = {
        user: [group[["press_time", "release_time", "key_code"]].to_numpy() for _, group in sessions.groupby("session_id")]
        for user, sessions in data.groupby("user_id")
    }
    all_users = [sessions for sessions in data_dict.values() if len(sessions) == SESSIONS_PER_USER]
    print(f"Users after filtering: {len(all_users)}")
    training_data = all_users[:-1050]
    validation_data = all_users[-1050:-1000]
    for dataset in (training_data, validation_data):
        for user_sequences in dataset:
            for index, sequence in enumerate(user_sequences):
                user_sequences[index] = _compute_features(sequence)

    def _save_pickle(path, obj):
        with open(path, "wb") as file:
            pickle.dump(obj, file)
                      
    _save_pickle(TRAINING_FEATURES_PICKLE, training_data)
    _save_pickle(VALIDATION_FEATURES_PICKLE, validation_data)
    return training_data, validation_data


class TrainDataset(Dataset):
    def __init__(self, data, batch_size, epoch_batch_count, seq_len):
        self.data = data
        self.batch_size = batch_size
        self.epoch_batch_count = epoch_batch_count
        self.seq_len = seq_len

    def __len__(self):
        return self.batch_size * self.epoch_batch_count

    def __getitem__(self, _):
        genuine_user, imposter_user = np.random.choice(len(self.data), size=2, replace=False)
        genuine_session_1, genuine_session_2 = np.random.choice(SESSIONS_PER_USER, size=2, replace=False)
        imposter_session = np.random.randint(SESSIONS_PER_USER)
        return tuple(
            _pad_sequence(self.data[user_index][session_index], self.seq_len)
            for user_index, session_index in [
                (genuine_user, genuine_session_1),
                (genuine_user, genuine_session_2),
                (imposter_user, imposter_session),
            ]
        )


class TestDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) * SESSIONS_PER_USER

    def __getitem__(self, idx):
        return _pad_sequence(self.data[idx // SESSIONS_PER_USER][idx % SESSIONS_PER_USER], self.seq_len)


def _make_model(feature_count, seq_len, target_len):
    return KeystrokeTransformer(6, feature_count, 20, 5, 10, seq_len, target_len, 0.1)


if __name__ == "__main__":
    config = Config().get_config_dict()
    gdown_id = config["preprocessed_data"]["aalto"]["keystroke"]
    if gdown_id and not CSV_PATH.exists():
        subprocess.run(f"gdown {gdown_id}", shell=True)

    training_data, validation_data = _load_data()

    hyperparams = config["hyperparams"]
    data_config = config["data"]
    batch_size = hyperparams["batch_size"]["aalto"]
    epoch_batch_count = hyperparams["epoch_batch_count"]["aalto"]
    seq_len = data_config["keystroke_sequence_len"]
    feature_count = hyperparams["keystroke_feature_count"]["aalto"]
    target_len = hyperparams["target_len"]
    learning_rate = hyperparams["learning_rate"]

    epochs = int(sys.argv[1])
    run_keystroke_training(
        project_root=PROJECT_ROOT,
        best_model_dir=BEST_MODELS_DIR,
        train_dataset=TrainDataset(training_data, batch_size, epoch_batch_count, seq_len),
        val_dataset=TestDataset(validation_data, seq_len),
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        model_factory=lambda: _make_model(feature_count, seq_len, target_len),
        compute_val_eer=lambda embeddings: Metric.cal_user_eer_aalto(
            embeddings.view(len(validation_data), SESSIONS_PER_USER, target_len),
            ENROLL_SESSIONS,
            VERIFY_SESSIONS,
        )[0],
    )
