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
from experiments.common.logger import get_logger
from experiments.common.lightning import run_keystroke_training

TRAINING_PICKLE = "training_keystroke_imu_data_all.pickle"
VALIDATION_PICKLE = "validation_keystroke_imu_data_all.pickle"
TESTING_PICKLE = "testing_keystroke_imu_data_all.pickle"
LOGGER = get_logger(__name__)


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
    compute_periodic_metrics: bool = False
    train_metrics_subset_users: int | None = None
    metrics_every_n_epochs: int = 5


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


def _copy_with_limited_sequences_per_session(data, limit: int):
    return [[session[:limit] for session in user] for user in data]


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


def _sample_user_subset(data, max_users: int | None, seed: int = 0):
    if max_users is None or len(data) <= max_users:
        return data

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(data), size=max_users, replace=False))
    return [data[index] for index in indices]


def _window_time(dataset_key: str, session) -> float:
    if dataset_key == "hmog":
        total = 0.0
        for sequence in session:
            keystroke = sequence[0]
            if total == 0:
                total += np.sum(keystroke, axis=0)[2] + keystroke[-1][0]
            else:
                total += np.sum(keystroke[-5:], axis=0)[2] + keystroke[-1][0]
        return float(total)

    if dataset_key == "humi":
        sequence = session[0][0]
        start = sequence[0][0]
        end = sequence[-1][0]
        index = -1
        while end == 0:
            end = sequence[index - 1][0]
            index -= 1
        return float((end - start) / 1000)

    raise ValueError(f"Unsupported nested metrics dataset: {dataset_key}")


def _make_nested_full_metrics_fn(
    raw_data,
    *,
    dataset_key: str,
    target_len: int,
    enrollment_sessions: int,
    verify_sessions: int,
):
    n_users = len(raw_data)

    def _get_periods(user_id: int):
        periods = [_window_time(dataset_key, raw_data[user_id][enrollment_sessions + j]) for j in range(verify_sessions)]
        for other_user in range(n_users):
            if other_user == user_id:
                continue
            for session_index in range(verify_sessions):
                periods.append(_window_time(dataset_key, raw_data[other_user][enrollment_sessions + session_index]))
        return periods

    def compute(flat_embeddings):
        embeddings = flat_embeddings.view(
            n_users,
            len(raw_data[0]),
            len(raw_data[0][0]),
            target_len,
        )
        acc_list, usab_list, tcr_list, fawi_list, frwi_list = [], [], [], [], []
        for user_id in range(n_users):
            enroll = embeddings[user_id, :enrollment_sessions].unsqueeze(0)
            genuine = embeddings[user_id, enrollment_sessions:]
            impostor = torch.cat(
                [
                    embeddings[:user_id, enrollment_sessions:].flatten(0, 1),
                    embeddings[user_id + 1 :, enrollment_sessions:].flatten(0, 1),
                ]
            )
            scores = Metric._get_distance_fn(dataset_key)(torch.cat([genuine, impostor]), enroll)
            labels = torch.tensor([1] * verify_sessions + [0] * ((n_users - 1) * verify_sessions))
            periods = _get_periods(user_id)
            acc, threshold = Metric.eer_compute(scores[:verify_sessions], scores[verify_sessions:])
            acc_list.append(acc)
            usab_list.append(Metric.calculate_usability(scores, threshold, periods, labels))
            tcr_list.append(Metric.calculate_TCR(scores, threshold, periods, labels))
            fawi_list.append(Metric.calculate_FAWI(scores, threshold, periods, labels))
            frwi_list.append(Metric.calculate_FRWI(scores, threshold, periods, labels))

        return {
            "eer": float(100 - np.mean(acc_list)),
            "usability": float(np.mean(usab_list)),
            "tcr": float(np.mean(tcr_list)),
            "fawi": float(np.mean(fawi_list)),
            "frwi": float(np.mean(frwi_list)),
        }

    return compute


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
    compute_val_metrics = None
    compute_train_metrics = None
    train_eval_dataset = None

    if spec.compute_periodic_metrics:
        train_metrics_source = _sample_user_subset(training_data, spec.train_metrics_subset_users)
        train_metrics_data = _copy_with_limited_sequences_per_session(
            train_metrics_source,
            spec.validation_sequences_per_session,
        )
        if train_metrics_source is training_data:
            LOGGER.info(
                "Using full training set for periodic %s metrics (%s users)",
                spec.dataset_key,
                len(training_data),
            )
        else:
            LOGGER.info(
                "Using %s sampled training users for periodic %s metrics (from %s total)",
                len(train_metrics_data),
                spec.dataset_key,
                len(training_data),
            )

        compute_val_metrics = _make_nested_full_metrics_fn(
            validation_data,
            dataset_key=spec.dataset_key,
            target_len=target_len,
            enrollment_sessions=enrollment_sessions,
            verify_sessions=verify_sessions,
        )
        compute_train_metrics = _make_nested_full_metrics_fn(
            train_metrics_data,
            dataset_key=spec.dataset_key,
            target_len=target_len,
            enrollment_sessions=enrollment_sessions,
            verify_sessions=verify_sessions,
        )
        train_eval_dataset = NestedEvalDataset(train_metrics_data, keystroke_columns=spec.keystroke_columns)

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
        compute_val_metrics=compute_val_metrics,
        compute_train_metrics=compute_train_metrics,
        train_eval_dataset=train_eval_dataset,
        metrics_every_n_epochs=spec.metrics_every_n_epochs,
        compute_val_eer=lambda embeddings: Metric.cal_user_eer(
            embeddings.view(len(validation_data), len(validation_data[0]), len(validation_data[0][0]), target_len),
            enrollment_sessions,
            verify_sessions,
            spec.dataset_key,
        )[0],
    )
