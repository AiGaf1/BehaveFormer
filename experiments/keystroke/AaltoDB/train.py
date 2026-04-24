import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

from Config import Config  # noqa: E402
from metrics import Metric  # noqa: E402

from experiments.common.datasets import EvalDataset, TrainDataset, feature_ranges, key_vocab_size, sample_user_subset  # noqa: E402
from experiments.common.lightning import run_keystroke_training  # noqa: E402
from experiments.common.logger import get_logger  # noqa: E402
from experiments.common.modeling import KeystrokeModel  # noqa: E402

LOGGER = get_logger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "AaltoDB" / "prep_data"
BEST_MODELS_DIR = HERE / "best_models"

CSV_PATH = DATA_DIR / "keystroke_data.csv"
TRAINING_FEATURES_PICKLE   = DATA_DIR / "training_features.pickle"
VALIDATION_FEATURES_PICKLE = DATA_DIR / "validation_features.pickle"
RAW_TRAINING_PICKLE        = DATA_DIR / "training_data.pickle"
RAW_VALIDATION_PICKLE      = DATA_DIR / "validation_data.pickle"

SESSIONS_PER_USER  = 15
ENROLL_SESSIONS    = 10
VERIFY_SESSIONS    = 5
TRAIN_METRICS_SUBSET_USERS = 256


def _compute_features(sequence):
    press, release, key = sequence.T
    features = np.zeros((len(sequence), 3), dtype=np.float32)
    features[:, 0] = (release - press) / 1000
    features[:-1, 1] = (press[1:] - release[:-1]) / 1000
    features[:, 2] = key + 1  # 0 stays reserved for padding
    return features


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _compute_dataset_features(raw_data):
    for user_sessions in raw_data:
        for idx, session in enumerate(user_sessions):
            user_sessions[idx] = _compute_features(session)
    return raw_data


def _load_data():
    if TRAINING_FEATURES_PICKLE.exists() and VALIDATION_FEATURES_PICKLE.exists():
        LOGGER.info("Using cached Aalto feature pickles")
        return _load_pickle(TRAINING_FEATURES_PICKLE), _load_pickle(VALIDATION_FEATURES_PICKLE)

    if RAW_TRAINING_PICKLE.exists() and RAW_VALIDATION_PICKLE.exists():
        LOGGER.info("Building Aalto feature pickles from cached raw pickles")
        training_data = _compute_dataset_features(_load_pickle(RAW_TRAINING_PICKLE))
        validation_data = _compute_dataset_features(_load_pickle(RAW_VALIDATION_PICKLE))
        _save_pickle(TRAINING_FEATURES_PICKLE, training_data)
        _save_pickle(VALIDATION_FEATURES_PICKLE, validation_data)
        return training_data, validation_data

    LOGGER.info("Building Aalto feature pickles")
    data = pd.read_csv(CSV_PATH)
    data_dict = {
        user: [group[["press_time", "release_time", "key_code"]].to_numpy() for _, group in sessions.groupby("session_id")]
        for user, sessions in data.groupby("user_id")
    }
    all_users = [sessions for sessions in data_dict.values() if len(sessions) == SESSIONS_PER_USER]
    LOGGER.info("Users after filtering: %s", len(all_users))

    training_data   = all_users[:-1050]
    validation_data = all_users[-1050:-1000]
    training_data = _compute_dataset_features(training_data)
    validation_data = _compute_dataset_features(validation_data)

    _save_pickle(TRAINING_FEATURES_PICKLE, training_data)
    _save_pickle(VALIDATION_FEATURES_PICKLE, validation_data)
    return training_data, validation_data


def _make_full_metrics_fn(raw_data, n_users, target_len):
    def _window_time(seq):
        return float(np.sum(seq[:, 0] + seq[:, 1]))

    all_periods = [
        [_window_time(raw_data[uid][ENROLL_SESSIONS + j]) for j in range(VERIFY_SESSIONS)]
        + [_window_time(raw_data[i][ENROLL_SESSIONS + j])
           for i in range(n_users) if i != uid
           for j in range(VERIFY_SESSIONS)]
        for uid in range(n_users)
    ]
    labels = torch.tensor([1] * VERIFY_SESSIONS + [0] * ((n_users - 1) * VERIFY_SESSIONS))

    def compute(flat_embeddings):
        embeddings = flat_embeddings.view(n_users, SESSIONS_PER_USER, target_len)
        enroll = embeddings[:, :ENROLL_SESSIONS]
        verify = embeddings[:, ENROLL_SESSIONS:]
        acc_list, usab_list, tcr_list, fawi_list, frwi_list = [], [], [], [], []
        for i in range(n_users):
            enroll_i = enroll[i].unsqueeze(0)
            impostors = torch.cat([verify[:i].flatten(0, 1), verify[i + 1:].flatten(0, 1)])
            all_seqs = torch.cat([verify[i], impostors]).unsqueeze(1)
            scores = torch.mean(torch.linalg.norm(all_seqs - enroll_i, dim=-1), dim=-1)
            acc, threshold = Metric.eer_compute(scores[:VERIFY_SESSIONS], scores[VERIFY_SESSIONS:])
            acc_list.append(acc)
            usab_list.append(Metric.calculate_usability(scores, threshold, all_periods[i], labels))
            tcr_list.append(Metric.calculate_TCR(scores, threshold, all_periods[i], labels))
            fawi_list.append(Metric.calculate_FAWI(scores, threshold, all_periods[i], labels))
            frwi_list.append(Metric.calculate_FRWI(scores, threshold, all_periods[i], labels))
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
    train_metrics_raw = sample_user_subset(training_data, TRAIN_METRICS_SUBSET_USERS)
    LOGGER.info("Using %s/%s training users for periodic train metrics", len(train_metrics_raw), len(training_data))

    hyperparams  = config["hyperparams"]
    data_config  = config["data"]
    batch_size        = int(os.getenv("BEHAVEFORMER_BATCH_SIZE", hyperparams["batch_size"]["aalto"]))
    epoch_batch_count = int(os.getenv("BEHAVEFORMER_EPOCH_BATCH_COUNT", hyperparams["epoch_batch_count"]["aalto"]))
    seq_len    = data_config["keystroke_sequence_len"]
    target_len = hyperparams["target_len"]
    lr         = hyperparams["learning_rate"]
    embed_dim  = hyperparams["key_embedding_dim"]
    vocab_size = key_vocab_size(training_data)
    ranges = feature_ranges(training_data)

    epochs                 = int(sys.argv[1])
    metrics_every_n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.getenv("BEHAVEFORMER_METRICS_EVERY_N_EPOCHS", 5))
    check_val_every_n_epoch = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.getenv("BEHAVEFORMER_VAL_EVERY_N_EPOCHS", 1))
    LOGGER.info("Aalto: batch=%s, batches/epoch=%s, val_every=%s, metrics_every=%s",
                batch_size, epoch_batch_count, check_val_every_n_epoch, metrics_every_n_epochs)

    run_keystroke_training(
        project_root=PROJECT_ROOT,
        best_model_dir=BEST_MODELS_DIR,
        train_dataset=TrainDataset(training_data, batch_size, epoch_batch_count, seq_len=seq_len),
        val_dataset=EvalDataset(validation_data, seq_len=seq_len),
        train_eval_dataset=EvalDataset(train_metrics_raw, seq_len=seq_len),
        batch_size=batch_size,
        learning_rate=lr,
        epochs=epochs,
        model_factory=lambda: KeystrokeModel(seq_len, target_len, vocab_size, embed_dim, ranges),
        compute_val_eer=lambda embeddings: Metric.cal_user_eer_aalto(
            embeddings.view(len(validation_data), SESSIONS_PER_USER, target_len),
            ENROLL_SESSIONS, VERIFY_SESSIONS,
        )[0],
        compute_val_metrics=_make_full_metrics_fn(validation_data, len(validation_data), target_len),
        compute_train_metrics=_make_full_metrics_fn(train_metrics_raw, len(train_metrics_raw), target_len),
        metrics_every_n_epochs=metrics_every_n_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
