import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BEST_MODELS_DIR = HERE / "best_models"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

from Config import Config  # noqa: E402
from metrics import Metric  # noqa: E402

from experiments.common.datasets import EvalDataset, KeystrokeData, TrainDataset, sample_user_subset  # noqa: E402
from experiments.common.lightning import run_keystroke_training  # noqa: E402
from experiments.common.logger import get_logger  # noqa: E402
from experiments.common.modeling import KeystrokeModel  # noqa: E402
from data.HMOGDB.prepare import ENROLL_SESSIONS, VERIFY_SESSIONS, load  # noqa: E402

LOGGER = get_logger(__name__)

DATASET_KEY = "hmog"
KEYSTROKE_COLUMNS = (0, 1, 2)  # hold_time, flight_time, key_code
VALIDATION_SEQUENCES_PER_SESSION = 50
TRAIN_METRICS_SUBSET_USERS = 256


def _session_seconds(session) -> float:
    total = 0.0
    for window_index, sequence in enumerate(session):
        keystrokes = sequence[0]
        flight_times = keystrokes[:, 2] if window_index == 0 else keystrokes[-5:, 2]
        total += float(np.sum(flight_times) + keystrokes[-1, 0])
    return total


def _make_metrics_fn(raw_data, *, target_len, enrollment_sessions, verify_sessions):
    n_users = len(raw_data)
    distance_fn = Metric._get_distance_fn(DATASET_KEY)
    labels = torch.tensor([1] * verify_sessions + [0] * ((n_users - 1) * verify_sessions))
    periods = [
        [_session_seconds(raw_data[uid][enrollment_sessions + j]) for j in range(verify_sessions)]
        + [_session_seconds(raw_data[other][enrollment_sessions + j])
           for other in range(n_users) if other != uid
           for j in range(verify_sessions)]
        for uid in range(n_users)
    ]

    def compute(flat_embeddings):
        embeddings = flat_embeddings.view(n_users, len(raw_data[0]), len(raw_data[0][0]), target_len)
        acc_list, usab_list, tcr_list, fawi_list, frwi_list = [], [], [], [], []
        for uid in range(n_users):
            enroll = embeddings[uid, :enrollment_sessions].unsqueeze(0)
            genuine = embeddings[uid, enrollment_sessions:]
            impostor = torch.cat([
                embeddings[:uid, enrollment_sessions:].flatten(0, 1),
                embeddings[uid + 1:, enrollment_sessions:].flatten(0, 1),
            ])
            scores = distance_fn(torch.cat([genuine, impostor]), enroll)
            acc, threshold = Metric.eer_compute(scores[:verify_sessions], scores[verify_sessions:])
            acc_list.append(acc)
            usab_list.append(Metric.calculate_usability(scores, threshold, periods[uid], labels))
            tcr_list.append(Metric.calculate_TCR(scores, threshold, periods[uid], labels))
            fawi_list.append(Metric.calculate_FAWI(scores, threshold, periods[uid], labels))
            frwi_list.append(Metric.calculate_FRWI(scores, threshold, periods[uid], labels))
        return {
            "eer":       float(100 - np.mean(acc_list)),
            "usability": float(np.mean(usab_list)),
            "tcr":       float(np.mean(tcr_list)),
            "fawi":      float(np.mean(fawi_list)),
            "frwi":      float(np.mean(frwi_list)),
        }

    return compute


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit("Epoch count argument is required")

    config = Config().get_config_dict()
    hyperparams = config["hyperparams"]
    data_config = config["data"]

    training_data, validation_data, vocab_size = load(
        config["preprocessed_data"][DATASET_KEY],
        VALIDATION_SEQUENCES_PER_SESSION,
        keystroke_sequence_len=data_config["keystroke_sequence_len"],
        imu_sequence_len=data_config["imu_sequence_len"],
        windowing_offset=config["data"]["hmog"]["windowing_offset"],
    )

    train_metrics_source = sample_user_subset(training_data, TRAIN_METRICS_SUBSET_USERS)
    train_metrics_data = [
        [session[:VALIDATION_SEQUENCES_PER_SESSION] for session in user]
        for user in train_metrics_source
    ]
    LOGGER.info("Using %s/%s training users for periodic HMOG metrics", len(train_metrics_data), len(training_data))

    batch_size          = hyperparams["batch_size"][DATASET_KEY]
    epoch_batch_count   = hyperparams["epoch_batch_count"][DATASET_KEY]
    seq_len             = data_config["keystroke_sequence_len"]
    embed_dim           = hyperparams["key_embedding_dim"]
    target_len          = hyperparams["target_len"]
    learning_rate       = hyperparams["learning_rate"]
    enrollment_sessions = hyperparams["number_of_enrollment_sessions"][DATASET_KEY]
    verify_sessions     = hyperparams["number_of_verify_sessions"][DATASET_KEY]

    keystroke_data = KeystrokeData(training_data, KEYSTROKE_COLUMNS)
    ranges = keystroke_data.feature_ranges()

    run_keystroke_training(
        project_root=PROJECT_ROOT,
        best_model_dir=BEST_MODELS_DIR,
        train_dataset=TrainDataset(training_data, batch_size, epoch_batch_count, columns=KEYSTROKE_COLUMNS),
        val_dataset=EvalDataset(validation_data, columns=KEYSTROKE_COLUMNS),
        train_eval_dataset=EvalDataset(train_metrics_data, columns=KEYSTROKE_COLUMNS),
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=int(sys.argv[1]),
        model_factory=lambda: KeystrokeModel(seq_len, target_len, vocab_size, embed_dim, ranges),
        compute_val_eer=lambda embeddings: Metric.cal_user_eer(
            embeddings.view(len(validation_data), len(validation_data[0]), len(validation_data[0][0]), target_len),
            enrollment_sessions, verify_sessions, DATASET_KEY,
        )[0],
        compute_val_metrics=_make_metrics_fn(
            validation_data, target_len=target_len,
            enrollment_sessions=enrollment_sessions, verify_sessions=verify_sessions,
        ),
        compute_train_metrics=_make_metrics_fn(
            train_metrics_data, target_len=target_len,
            enrollment_sessions=enrollment_sessions, verify_sessions=verify_sessions,
        ),
        metrics_every_n_epochs=5,
    )
