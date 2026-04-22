import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BEST_MODELS_DIR = HERE / "best_models"
CHECKPOINTS_DIR = HERE / "checkpoints"
PREP_DATA_DIR = PROJECT_ROOT / "data" / "HMOGDB" / "prep_data"

TRAINING_PICKLE = "training_keystroke_imu_data_all.pickle"
VALIDATION_PICKLE = "validation_keystroke_imu_data_all.pickle"
TESTING_PICKLE = "testing_keystroke_imu_data_all.pickle"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

from Config import Config
from metrics import Metric
from model import Model
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


def _pickle_path(name: str) -> Path:
    prep_path = PREP_DATA_DIR / name
    root_path = PROJECT_ROOT / name
    if prep_path.exists():
        return prep_path
    if root_path.exists():
        return root_path
    return prep_path


def _maybe_download(file_id: str, target_path: Path) -> None:
    if target_path.exists() or not file_id:
        return
    subprocess.run(f"gdown {file_id}", shell=True, check=True, cwd=target_path.parent)


def scale(data):
    for user in data:
        for session in user:
            for index in range(len(session)):
                keystroke = session[index][0].astype(np.float64, copy=True)
                imu = session[index][1].astype(np.float64, copy=True)

                for feature in range(10):
                    if feature == 9:
                        keystroke[:, feature] = keystroke[:, feature] / 255
                    else:
                        keystroke[:, feature] = keystroke[:, feature] / 1000

                for feature in range(36):
                    if feature in [0, 1, 2]:
                        imu[:, feature] = imu[:, feature] / 10
                    elif feature in [3, 4, 5, 15, 16, 17]:
                        imu[:, feature] = imu[:, feature] / 1000
                    elif feature in [24, 25, 26]:
                        imu[:, feature] = imu[:, feature] / 100
                    elif feature in [27, 28, 29]:
                        imu[:, feature] = imu[:, feature] / 10000

                session[index][0] = keystroke
                session[index][1] = imu


class TrainDataset(Dataset):
    def __init__(self, training_data, batch_size, epoch_batch_count):
        self.training_data = training_data
        self.batch_size = batch_size
        self.epoch_batch_count = epoch_batch_count

    def __len__(self):
        return self.batch_size * self.epoch_batch_count

    def __getitem__(self, _):
        genuine_user_idx = np.random.randint(0, len(self.training_data))
        imposter_user_idx = np.random.randint(0, len(self.training_data))
        while imposter_user_idx == genuine_user_idx:
            imposter_user_idx = np.random.randint(0, len(self.training_data))

        genuine_sess_1 = np.random.randint(0, len(self.training_data[0]))
        genuine_sess_2 = np.random.randint(0, len(self.training_data[0]))
        while genuine_sess_2 == genuine_sess_1:
            genuine_sess_2 = np.random.randint(0, len(self.training_data[0]))
        imposter_sess = np.random.randint(0, len(self.training_data[0]))

        genuine_seq_1 = np.random.randint(0, len(self.training_data[genuine_user_idx][genuine_sess_1]))
        genuine_seq_2 = np.random.randint(0, len(self.training_data[genuine_user_idx][genuine_sess_2]))
        imposter_seq = np.random.randint(0, len(self.training_data[imposter_user_idx][imposter_sess]))

        anchor = self.training_data[genuine_user_idx][genuine_sess_1][genuine_seq_1][0]
        positive = self.training_data[genuine_user_idx][genuine_sess_2][genuine_seq_2][0]
        negative = self.training_data[imposter_user_idx][imposter_sess][imposter_seq][0]
        return anchor, positive, negative


class TestDataset(Dataset):
    def __init__(self, eval_data):
        self.eval_data = eval_data
        self.num_sessions = len(self.eval_data[0])
        self.num_seqs = len(self.eval_data[0][0])

    def __len__(self):
        return math.ceil(len(self.eval_data) * self.num_sessions * self.num_seqs)

    def __getitem__(self, idx):
        total_session = idx // self.num_seqs
        user_idx = total_session // self.num_sessions
        session_idx = total_session % self.num_sessions
        seq_idx = idx % self.num_seqs
        return self.eval_data[user_idx][session_idx][seq_idx][0]


def _make_model(feature_count, seq_len, target_len):
    return Model(feature_count, seq_len, target_len)


if __name__ == "__main__":
    configure_lightning_environment()

    config = Config().get_config_dict()
    hyperparams = config["hyperparams"]
    data_config = config["data"]
    preprocessed_data = config["preprocessed_data"]["hmog"]

    training_path = _pickle_path(TRAINING_PICKLE)
    validation_path = _pickle_path(VALIDATION_PICKLE)
    testing_path = _pickle_path(TESTING_PICKLE)
    _maybe_download(preprocessed_data["train"], training_path)
    _maybe_download(preprocessed_data["val"], validation_path)
    _maybe_download(preprocessed_data["test"], testing_path)

    with open(training_path, "rb") as file:
        training_data = pickle.load(file)
    with open(validation_path, "rb") as file:
        validation_data = pickle.load(file)

    for user in validation_data:
        for index, session in enumerate(user):
            user[index] = session[:50]

    scale(training_data)
    scale(validation_data)

    batch_size = hyperparams["batch_size"]["hmog"]
    epoch_batch_count = hyperparams["epoch_batch_count"]["hmog"]
    seq_len = data_config["keystroke_sequence_len"]
    feature_count = hyperparams["keystroke_feature_count"]["hmog"]
    target_len = hyperparams["target_len"]
    learning_rate = hyperparams["learning_rate"]
    enrollment_sessions = hyperparams["number_of_enrollment_sessions"]["hmog"]
    verify_sessions = hyperparams["number_of_verify_sessions"]["hmog"]

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
        compute_val_eer=lambda embeddings: Metric.cal_user_eer(
            embeddings.view(len(validation_data), len(validation_data[0]), len(validation_data[0][0]), target_len),
            enrollment_sessions,
            verify_sessions,
            "hmog",
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
            dataset_name="hmog",
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
            TrainDataset(training_data, batch_size, epoch_batch_count),
            TestDataset(validation_data),
            batch_size=batch_size,
            num_workers=recommended_num_workers(),
            pin_memory=use_gpu,
        ),
    )
