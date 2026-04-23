import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

from Config import Config
from metrics import Metric
from model import KeystrokeTransformer

from experiments.common.lightning import run_keystroke_training
from experiments.common.logger import get_logger

LOGGER = get_logger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "AaltoDB" / "prep_data"
BEST_MODELS_DIR = HERE / "best_models"

CSV_PATH = DATA_DIR / "keystroke_data.csv"
TRAINING_FEATURES_PICKLE = DATA_DIR / "training_features.pickle"
VALIDATION_FEATURES_PICKLE = DATA_DIR / "validation_features.pickle"

SESSIONS_PER_USER = 15
ENROLL_SESSIONS = 10
VERIFY_SESSIONS = 5
TRAIN_METRICS_SUBSET_USERS = 256


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
        LOGGER.info("Using cached Aalto feature pickles")
        return _load_pickle(TRAINING_FEATURES_PICKLE), _load_pickle(VALIDATION_FEATURES_PICKLE)

    LOGGER.info("Building Aalto feature pickles")
    data = pd.read_csv(CSV_PATH)
    assert not data.isnull().values.any()
    data_dict = {
        user: [group[["press_time", "release_time", "key_code"]].to_numpy() for _, group in sessions.groupby("session_id")]
        for user, sessions in data.groupby("user_id")
    }
    all_users = [sessions for sessions in data_dict.values() if len(sessions) == SESSIONS_PER_USER]
    LOGGER.info("Users after filtering: %s", len(all_users))
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


def _sample_user_subset(data, max_users, seed=0):
    if len(data) <= max_users:
        return data

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(data), size=max_users, replace=False))
    return [data[index] for index in indices]


def _make_full_metrics_fn(raw_data, n_users, target_len):
    """Returns a callable that takes flat embeddings and returns a metrics dict."""
    import torch

    def _window_time(seq):
        return np.sum(seq, axis=0)[2] + seq[-1][0]

    def _get_periods(user_id):
        periods = [_window_time(raw_data[user_id][ENROLL_SESSIONS + j]) for j in range(VERIFY_SESSIONS)]
        periods += [_window_time(raw_data[i][ENROLL_SESSIONS]) for i in range(n_users) if i != user_id]
        return periods

    def _scores_all(embeddings):
        enroll = embeddings[:, :ENROLL_SESSIONS]
        verify = embeddings[:, ENROLL_SESSIONS:]
        scores_list = []
        for i in range(n_users):
            enroll_i = enroll[i].unsqueeze(0)
            impostors = torch.cat([verify[:i].flatten(0, 1), verify[i+1:].flatten(0, 1)], dim=0)
            all_seqs = torch.cat([verify[i], impostors], dim=0).unsqueeze(1)
            scores_list.append(torch.mean(torch.linalg.norm(all_seqs - enroll_i, dim=-1), dim=-1))
        return scores_list

    def compute(flat_embeddings):
        embeddings = flat_embeddings.view(n_users, SESSIONS_PER_USER, target_len)
        all_scores = _scores_all(embeddings)
        acc_list, usab_list, tcr_list, fawi_list, frwi_list = [], [], [], [], []
        for i, scores in enumerate(all_scores):
            labels = torch.tensor([1] * VERIFY_SESSIONS + [0] * (n_users - 1))
            periods = _get_periods(i)
            acc, threshold = Metric.eer_compute(scores[:VERIFY_SESSIONS], scores[VERIFY_SESSIONS:])
            acc_list.append(acc)
            usab_list.append(Metric.calculate_usability(scores, threshold, periods, labels))
            tcr_list.append(Metric.calculate_TCR(scores, threshold, periods, labels))
            fawi_list.append(Metric.calculate_FAWI(scores, threshold, periods, labels))
            frwi_list.append(Metric.calculate_FRWI(scores, threshold, periods, labels))
        return {
            "eer":       float(100 - np.mean(acc_list)),
            "usability": float(np.mean(usab_list)),
            "tcr":       float(np.mean(tcr_list)),
            "fawi":      float(np.mean(fawi_list)),
            "frwi":      float(np.mean(frwi_list)),
        }

    return compute

if __name__ == "__main__":
    config = Config().get_config_dict()
    gdown_id = config["preprocessed_data"]["aalto"]["keystroke"]
    if gdown_id and not CSV_PATH.exists():
        subprocess.run(f"gdown {gdown_id}", shell=True)

    training_data, validation_data = _load_data()
    train_metrics_data = _sample_user_subset(training_data, TRAIN_METRICS_SUBSET_USERS)
    LOGGER.info(
        "Using %s sampled training users for periodic train metrics (from %s total)",
        len(train_metrics_data),
        len(training_data),
    )

    hyperparams = config["hyperparams"]
    data_config = config["data"]
    batch_size = hyperparams["batch_size"]["aalto"]
    epoch_batch_count = hyperparams["epoch_batch_count"]["aalto"]
    seq_len = data_config["keystroke_sequence_len"]
    feature_count = hyperparams["keystroke_feature_count"]["aalto"]
    target_len = hyperparams["target_len"]
    learning_rate = hyperparams["learning_rate"]

    epochs = int(sys.argv[1])
    metrics_every_n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    run_keystroke_training(
        project_root=PROJECT_ROOT,
        best_model_dir=BEST_MODELS_DIR,
        train_dataset=TrainDataset(training_data, batch_size, epoch_batch_count, seq_len),
        val_dataset=TestDataset(validation_data, seq_len),
        train_eval_dataset=TestDataset(train_metrics_data, seq_len),
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        model_factory=lambda: _make_model(feature_count, seq_len, target_len),
        compute_val_eer=lambda embeddings: Metric.cal_user_eer_aalto(
            embeddings.view(len(validation_data), SESSIONS_PER_USER, target_len),
            ENROLL_SESSIONS,
            VERIFY_SESSIONS,
        )[0],
        compute_val_metrics=_make_full_metrics_fn(validation_data, len(validation_data), target_len),
        compute_train_metrics=_make_full_metrics_fn(train_metrics_data, len(train_metrics_data), target_len),
        metrics_every_n_epochs=metrics_every_n_epochs,
    )
