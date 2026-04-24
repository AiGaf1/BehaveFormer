import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from Config import Config
from metrics import Metric

from experiments.common.datasets import EvalDataset, TrainDataset, feature_ranges, key_vocab_size, sample_user_subset, scale_features
from experiments.common.lightning import run_keystroke_training
from experiments.common.logger import get_logger

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
    train_metrics_subset_users: int | None = None
    metrics_every_n_epochs: int = 5


def _pickle_path(project_root: Path, dataset_dir_name: str, name: str) -> Path:
    prep_path = project_root / "data" / dataset_dir_name / "prep_data" / name
    root_path = project_root / name
    return prep_path if prep_path.exists() or not root_path.exists() else root_path


def _maybe_download(file_id: str, target_path: Path) -> None:
    if target_path.exists() or not file_id:
        return
    subprocess.run(f"gdown {file_id}", shell=True, check=True, cwd=target_path.parent)


def _window_time(dataset_key: str, session) -> float:
    if dataset_key == "hmog":
        total = 0.0
        for sequence in session:
            ks = sequence[0]
            total += np.sum(ks, axis=0)[2] + ks[-1][0] if total == 0 else np.sum(ks[-5:], axis=0)[2] + ks[-1][0]
        return float(total)

    if dataset_key == "humi":
        seq = session[0][0]
        start = seq[0][0]
        end = seq[-1][0]
        i = -1
        while end == 0:
            end = seq[i - 1][0]
            i -= 1
        return float((end - start) / 1000)

    raise ValueError(f"Unsupported nested metrics dataset: {dataset_key}")


def _make_nested_full_metrics_fn(raw_data, *, dataset_key, target_len, enrollment_sessions, verify_sessions):
    n_users = len(raw_data)
    distance_fn = Metric._get_distance_fn(dataset_key)
    labels = torch.tensor([1] * verify_sessions + [0] * ((n_users - 1) * verify_sessions))
    periods = [
        [_window_time(dataset_key, raw_data[user_id][enrollment_sessions + j]) for j in range(verify_sessions)]
        + [
            _window_time(dataset_key, raw_data[other][enrollment_sessions + j])
            for other in range(n_users)
            if other != user_id
            for j in range(verify_sessions)
        ]
        for user_id in range(n_users)
    ]

    def compute(flat_embeddings):
        embeddings = flat_embeddings.view(n_users, len(raw_data[0]), len(raw_data[0][0]), target_len)
        acc_list, usab_list, tcr_list, fawi_list, frwi_list = [], [], [], [], []
        for user_id in range(n_users):
            enroll = embeddings[user_id, :enrollment_sessions].unsqueeze(0)
            genuine = embeddings[user_id, enrollment_sessions:]

            impostor = torch.cat([
                embeddings[:user_id, enrollment_sessions:].flatten(0, 1),
                embeddings[user_id + 1:, enrollment_sessions:].flatten(0, 1),
            ])
            scores = distance_fn(torch.cat([genuine, impostor]), enroll)
            acc, threshold = Metric.eer_compute(scores[:verify_sessions], scores[verify_sessions:])
            acc_list.append(acc)
            usab_list.append(Metric.calculate_usability(scores, threshold, periods[user_id], labels))
            tcr_list.append(Metric.calculate_TCR(scores, threshold, periods[user_id], labels))
            fawi_list.append(Metric.calculate_FAWI(scores, threshold, periods[user_id], labels))
            frwi_list.append(Metric.calculate_FRWI(scores, threshold, periods[user_id], labels))
        return {
            "eer":       float(100 - np.mean(acc_list)),
            "usability": float(np.mean(usab_list)),
            "tcr":       float(np.mean(tcr_list)),
            "fawi":      float(np.mean(fawi_list)),
            "frwi":      float(np.mean(frwi_list)),
        }

    return compute


def run_nested_keystroke_training_script(*, spec: NestedKeystrokeDatasetSpec, project_root: Path, best_model_dir: Path, model_factory, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("Epoch count argument is required")

    config = Config().get_config_dict()
    hyperparams = config["hyperparams"]
    data_config = config["data"]
    preprocessed_data = config["preprocessed_data"][spec.dataset_key]

    training_path   = _pickle_path(project_root, spec.dataset_dir_name, TRAINING_PICKLE)
    validation_path = _pickle_path(project_root, spec.dataset_dir_name, VALIDATION_PICKLE)
    _maybe_download(preprocessed_data["train"], training_path)
    _maybe_download(preprocessed_data["val"],   validation_path)
    _maybe_download(preprocessed_data["test"],  _pickle_path(project_root, spec.dataset_dir_name, TESTING_PICKLE))

    with open(training_path, "rb") as f:
        training_data = pickle.load(f)
    with open(validation_path, "rb") as f:
        validation_data = pickle.load(f)

    for user in validation_data:
        for idx, session in enumerate(user):
            user[idx] = session[:spec.validation_sequences_per_session]

    scale_features(training_data,  spec.keystroke_scale_map, spec.imu_scale_map)
    scale_features(validation_data, spec.keystroke_scale_map, spec.imu_scale_map)

    batch_size          = hyperparams["batch_size"][spec.dataset_key]
    epoch_batch_count   = hyperparams["epoch_batch_count"][spec.dataset_key]
    seq_len             = data_config["keystroke_sequence_len"]
    embed_dim           = hyperparams["key_embedding_dim"]
    target_len          = hyperparams["target_len"]
    learning_rate       = hyperparams["learning_rate"]
    enrollment_sessions = hyperparams["number_of_enrollment_sessions"][spec.dataset_key]
    verify_sessions     = hyperparams["number_of_verify_sessions"][spec.dataset_key]

    columns = spec.keystroke_columns
    vocab_size = key_vocab_size(training_data, columns)
    ranges = feature_ranges(training_data, columns)

    compute_val_metrics = compute_train_metrics = train_eval_dataset = None
    if spec.train_metrics_subset_users is not None:
        train_metrics_source = sample_user_subset(training_data, spec.train_metrics_subset_users)
        train_metrics_data = [[session[:spec.validation_sequences_per_session] for session in user] for user in train_metrics_source]
        LOGGER.info("Using %s/%s training users for periodic %s metrics", len(train_metrics_data), len(training_data), spec.dataset_key)
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
        train_eval_dataset = EvalDataset(train_metrics_data, columns=columns)

    run_keystroke_training(
        project_root=project_root,
        best_model_dir=best_model_dir,
        train_dataset=TrainDataset(training_data, batch_size, epoch_batch_count, columns=columns),
        val_dataset=EvalDataset(validation_data, columns=columns),
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=int(argv[0]),
        model_factory=lambda: model_factory(seq_len, target_len, vocab_size, embed_dim, ranges),
        compute_val_metrics=compute_val_metrics,
        compute_train_metrics=compute_train_metrics,
        train_eval_dataset=train_eval_dataset,
        metrics_every_n_epochs=spec.metrics_every_n_epochs,
        compute_val_eer=lambda embeddings: Metric.cal_user_eer(
            embeddings.view(len(validation_data), len(validation_data[0]), len(validation_data[0][0]), target_len),
            enrollment_sessions, verify_sessions, spec.dataset_key,
        )[0],
    )
