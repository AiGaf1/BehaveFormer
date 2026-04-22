import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "AaltoDB" / "prep_data"
BEST_MODELS_DIR = HERE / "best_models"
CHECKPOINTS_DIR = HERE / "checkpoints"

CSV_PATH = DATA_DIR / "keystroke_data.csv"
TRAINING_PICKLE = DATA_DIR / "training_data.pickle"
VALIDATION_PICKLE = DATA_DIR / "validation_data.pickle"
TESTING_PICKLE = DATA_DIR / "testing_data.pickle"
TRAINING_FEATURES_PICKLE = DATA_DIR / "training_features.pickle"
VALIDATION_FEATURES_PICKLE = DATA_DIR / "validation_features.pickle"

SESSIONS_PER_USER = 15
ENROLL_SESSIONS = 10
VERIFY_SESSIONS = 5

for path in ["utils", "evaluation", ""]:
    sys.path.append(str(PROJECT_ROOT / path))

from Config import Config
from metrics import Metric
from model import KeystrokeTransformer
from experiments.keystroke.common.lightning import (
    KeystrokeDataModule,
    KeystrokeLightningModule,
    TrainingArtifactsCallback,
    build_trainer,
    configure_lightning_environment,
    load_resume_state,
    recommended_num_workers,
    setup_wandb,
)


def _save_pickle(path, obj):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def _load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _pad_sequence(sequence, seq_len):
    if len(sequence) == seq_len:
        return sequence
    if len(sequence) < seq_len:
        padding = np.zeros((seq_len - len(sequence), sequence.shape[1]))
        return np.vstack([sequence, padding])
    return sequence[:seq_len]


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


def preprocess():
    data = pd.read_csv(CSV_PATH)
    assert not data.isnull().values.any()
    data_dict = {
        user: [group[["press_time", "release_time", "key_code"]].to_numpy() for _, group in sessions.groupby("session_id")]
        for user, sessions in data.groupby("user_id")
    }
    all_users = [sessions for sessions in data_dict.values() if len(sessions) == SESSIONS_PER_USER]
    print(f"Users after filtering: {len(all_users)}")
    for pickle_path, split in zip(
        [TRAINING_PICKLE, VALIDATION_PICKLE, TESTING_PICKLE],
        [all_users[:-1050], all_users[-1050:-1000], all_users[-1000:]],
    ):
        _save_pickle(pickle_path, split)


def _ensure_preprocessed_pickles():
    if all(path.exists() for path in [TRAINING_PICKLE, VALIDATION_PICKLE, TESTING_PICKLE]):
        print("Using cached Aalto split pickles")
        return
    print("Building Aalto split pickles")
    preprocess()


def _load_feature_pickles():
    if TRAINING_FEATURES_PICKLE.exists() and VALIDATION_FEATURES_PICKLE.exists():
        print("Using cached Aalto feature pickles")
        return _load_pickle(TRAINING_FEATURES_PICKLE), _load_pickle(VALIDATION_FEATURES_PICKLE)

    print("Building Aalto feature pickles")
    training_data = _load_pickle(TRAINING_PICKLE)
    validation_data = _load_pickle(VALIDATION_PICKLE)
    for dataset in (training_data, validation_data):
        for user_sequences in dataset:
            for index, sequence in enumerate(user_sequences):
                user_sequences[index] = _compute_features(sequence)
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
    configure_lightning_environment()

    config = Config().get_config_dict()
    gdown_id = config["preprocessed_data"]["aalto"]["keystroke"]
    if gdown_id and not CSV_PATH.exists():
        subprocess.run(f"gdown {gdown_id}", shell=True)

    _ensure_preprocessed_pickles()
    training_data, validation_data = _load_feature_pickles()

    hyperparams = config["hyperparams"]
    data_config = config["data"]
    batch_size = hyperparams["batch_size"]["aalto"]
    epoch_batch_count = hyperparams["epoch_batch_count"]["aalto"]
    seq_len = data_config["keystroke_sequence_len"]
    feature_count = hyperparams["keystroke_feature_count"]["aalto"]
    target_len = hyperparams["target_len"]
    learning_rate = hyperparams["learning_rate"]

    BEST_MODELS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    epochs = int(sys.argv[1])
    epoch_offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    best_eer, model_state, optimizer_state = load_resume_state(CHECKPOINTS_DIR, epoch_offset)

    use_gpu = config["GPU"] == "True" and torch.cuda.is_available()
    if use_gpu:
        torch.set_float32_matmul_precision("high")

    module = KeystrokeLightningModule(
        model_factory=lambda: _make_model(feature_count, seq_len, target_len),
        learning_rate=learning_rate,
        compute_val_eer=lambda embeddings: Metric.cal_user_eer_aalto(
            embeddings.view(len(validation_data), SESSIONS_PER_USER, target_len),
            ENROLL_SESSIONS,
            VERIFY_SESSIONS,
        )[0],
    )
    if model_state is not None:
        module.model.load_state_dict(model_state)
        module.resume_optimizer_state = optimizer_state

    wandb_logger = setup_wandb(PROJECT_ROOT, module.model)
    trainer = build_trainer(
        use_gpu,
        epochs,
        TrainingArtifactsCallback(
            BEST_MODELS_DIR,
            CHECKPOINTS_DIR,
            dataset_name="aalto",
            epoch_offset=epoch_offset,
            best_eer=best_eer,
            wandb_logger=wandb_logger,
            resume_from_epoch=epoch_offset or None,
        ),
        wandb_logger,
    )
    trainer.fit(
        module,
        datamodule=KeystrokeDataModule(
            TrainDataset(training_data, batch_size, epoch_batch_count, seq_len),
            TestDataset(validation_data, seq_len),
            batch_size=batch_size,
            num_workers=recommended_num_workers(),
            pin_memory=use_gpu,
        ),
    )
