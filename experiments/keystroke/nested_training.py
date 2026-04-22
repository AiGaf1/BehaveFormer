import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Config import Config
from metrics import Metric
from experiments.keystroke.common.lightning import run_keystroke_training

TRAINING_PICKLE = "training_keystroke_imu_data_all.pickle"
VALIDATION_PICKLE = "validation_keystroke_imu_data_all.pickle"
TESTING_PICKLE = "testing_keystroke_imu_data_all.pickle"


@dataclass(frozen=True)
class NestedKeystrokeDatasetSpec:
    dataset_key: str
    dataset_dir_name: str
    validation_sequences_per_session: int
    keystroke_scale_map: dict[int, float]
    imu_scale_map: dict[int, float]
    keystroke_columns: tuple[int, ...] | None = None
    convert_to_float: bool = False
    loss_fn: object = None


def _pickle_path(project_root: Path, dataset_dir_name: str, name: str) -> Path:
    prep_path = project_root / "data" / dataset_dir_name / "prep_data" / name
    root_path = project_root / name
    if prep_path.exists():
        return prep_path
    if root_path.exists():
        return root_path
    return prep_path


def _maybe_download(file_id: str, target_path: Path) -> None:
    if target_path.exists() or not file_id:
        return
    subprocess.run(f"gdown {file_id}", shell=True, check=True, cwd=target_path.parent)


def convert_nested_modalities_to_float(data) -> None:
    for user in data:
        for session in user:
            for sequence in session:
                for index in [0, 1]:
                    if sequence[index].dtype != np.float64:
                        sequence[index] = sequence[index].astype(np.float64)


def limit_sequences_per_session(data, limit: int) -> None:
    for user in data:
        for index, session in enumerate(user):
            user[index] = session[:limit]


def scale_nested_modalities(data, keystroke_scale_map: dict[int, float], imu_scale_map: dict[int, float]) -> None:
    for user in data:
        for session in user:
            for index, sequence in enumerate(session):
                keystroke = sequence[0].astype(np.float64, copy=True)
                imu = sequence[1].astype(np.float64, copy=True)

                for feature, divisor in keystroke_scale_map.items():
                    keystroke[:, feature] = keystroke[:, feature] / divisor

                for feature, divisor in imu_scale_map.items():
                    imu[:, feature] = imu[:, feature] / divisor

                session[index][0] = keystroke
                session[index][1] = imu


class NestedTripletDataset(Dataset):
    def __init__(self, training_data, batch_size, epoch_batch_count, keystroke_columns=None):
        self.training_data = training_data
        self.batch_size = batch_size
        self.epoch_batch_count = epoch_batch_count
        self.keystroke_columns = keystroke_columns

    def __len__(self):
        return self.batch_size * self.epoch_batch_count

    def __getitem__(self, _):
        genuine_user_idx, imposter_user_idx = np.random.choice(len(self.training_data), size=2, replace=False)
        genuine_sess_1, genuine_sess_2 = np.random.choice(
            len(self.training_data[genuine_user_idx]),
            size=2,
            replace=False,
        )
        imposter_sess = np.random.randint(len(self.training_data[imposter_user_idx]))

        genuine_seq_1 = np.random.randint(len(self.training_data[genuine_user_idx][genuine_sess_1]))
        genuine_seq_2 = np.random.randint(len(self.training_data[genuine_user_idx][genuine_sess_2]))
        imposter_seq = np.random.randint(len(self.training_data[imposter_user_idx][imposter_sess]))

        return (
            self._select_keystroke(self.training_data[genuine_user_idx][genuine_sess_1][genuine_seq_1]),
            self._select_keystroke(self.training_data[genuine_user_idx][genuine_sess_2][genuine_seq_2]),
            self._select_keystroke(self.training_data[imposter_user_idx][imposter_sess][imposter_seq]),
        )

    def _select_keystroke(self, sequence):
        keystroke = sequence[0]
        if self.keystroke_columns is None:
            return keystroke
        return keystroke[:, self.keystroke_columns]


class NestedEvalDataset(Dataset):
    def __init__(self, eval_data, keystroke_columns=None):
        self.eval_data = eval_data
        self.keystroke_columns = keystroke_columns
        self.num_sessions = len(self.eval_data[0])
        self.num_sequences = len(self.eval_data[0][0])

    def __len__(self):
        return len(self.eval_data) * self.num_sessions * self.num_sequences

    def __getitem__(self, idx):
        total_session = idx // self.num_sequences
        user_idx = total_session // self.num_sessions
        session_idx = total_session % self.num_sessions
        seq_idx = idx % self.num_sequences
        return self._select_keystroke(self.eval_data[user_idx][session_idx][seq_idx])

    def _select_keystroke(self, sequence):
        keystroke = sequence[0]
        if self.keystroke_columns is None:
            return keystroke
        return keystroke[:, self.keystroke_columns]


def run_nested_keystroke_training_script(
    *,
    spec: NestedKeystrokeDatasetSpec,
    project_root: Path,
    best_model_dir: Path,
    model_factory,
    argv=None,
):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("Epoch count argument is required")

    config = Config().get_config_dict()
    hyperparams = config["hyperparams"]
    data_config = config["data"]
    preprocessed_data = config["preprocessed_data"][spec.dataset_key]

    training_path = _pickle_path(project_root, spec.dataset_dir_name, TRAINING_PICKLE)
    validation_path = _pickle_path(project_root, spec.dataset_dir_name, VALIDATION_PICKLE)
    _maybe_download(preprocessed_data["train"], training_path)
    _maybe_download(preprocessed_data["val"], validation_path)
    _maybe_download(preprocessed_data["test"], _pickle_path(project_root, spec.dataset_dir_name, TESTING_PICKLE))

    with open(training_path, "rb") as file:
        training_data = pickle.load(file)
    with open(validation_path, "rb") as file:
        validation_data = pickle.load(file)

    if spec.convert_to_float:
        convert_nested_modalities_to_float(training_data)
        convert_nested_modalities_to_float(validation_data)

    limit_sequences_per_session(validation_data, spec.validation_sequences_per_session)
    scale_nested_modalities(training_data, spec.keystroke_scale_map, spec.imu_scale_map)
    scale_nested_modalities(validation_data, spec.keystroke_scale_map, spec.imu_scale_map)

    batch_size = hyperparams["batch_size"][spec.dataset_key]
    epoch_batch_count = hyperparams["epoch_batch_count"][spec.dataset_key]
    seq_len = data_config["keystroke_sequence_len"]
    feature_count = hyperparams["keystroke_feature_count"][spec.dataset_key]
    target_len = hyperparams["target_len"]
    learning_rate = hyperparams["learning_rate"]
    enrollment_sessions = hyperparams["number_of_enrollment_sessions"][spec.dataset_key]
    verify_sessions = hyperparams["number_of_verify_sessions"][spec.dataset_key]

    epochs = int(argv[0])
    run_keystroke_training(
        project_root=project_root,
        best_model_dir=best_model_dir,
        train_dataset=NestedTripletDataset(
            training_data,
            batch_size,
            epoch_batch_count,
            keystroke_columns=spec.keystroke_columns,
        ),
        val_dataset=NestedEvalDataset(validation_data, keystroke_columns=spec.keystroke_columns),
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        model_factory=lambda: model_factory(feature_count, seq_len, target_len),
        loss_fn=spec.loss_fn,
        compute_val_eer=lambda embeddings: Metric.cal_user_eer(
            embeddings.view(len(validation_data), len(validation_data[0]), len(validation_data[0][0]), target_len),
            enrollment_sessions,
            verify_sessions,
            spec.dataset_key,
        )[0],
    )
