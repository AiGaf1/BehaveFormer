import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

# isort: split
from Config import Config  # noqa: E402
from metrics import Metric  # noqa: E402

from data.AaltoDB.build import maybe_download  # noqa: E402
from data.AaltoDB.prepare import ENROLL_SESSIONS, SESSIONS_PER_USER, VERIFY_SESSIONS, load  # noqa: E402
from experiments.common.datasets import EvalDataset, KeystrokeData, TrainDataset, sample_user_subset  # noqa: E402
from experiments.common.lightning import run_keystroke_training  # noqa: E402
from experiments.common.logger import get_logger  # noqa: E402
from experiments.common.modeling import KeystrokeModel  # noqa: E402

LOGGER = get_logger(__name__)
BEST_MODELS_DIR = HERE / "best_models"
TRAIN_METRICS_SUBSET_USERS = 256


def _make_metrics_fn(raw_data, target_len):
    n_users = len(raw_data)

    def _session_seconds(seq):
        return float(np.sum(seq[:, 0] + seq[:, 1]))

    all_periods = [
        [_session_seconds(raw_data[uid][ENROLL_SESSIONS + j]) for j in range(VERIFY_SESSIONS)]
        + [_session_seconds(raw_data[i][ENROLL_SESSIONS + j])
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
    maybe_download(config["preprocessed_data"]["aalto"]["keystroke"])

    training_data, validation_data = load()
    train_metrics_raw = sample_user_subset(training_data, TRAIN_METRICS_SUBSET_USERS)
    LOGGER.info("Using %s/%s training users for periodic train metrics", len(train_metrics_raw), len(training_data))

    hyperparams = config["hyperparams"]
    data_config = config["data"]
    batch_size        = hyperparams["batch_size"]["aalto"]
    epoch_batch_count = hyperparams["epoch_batch_count"]["aalto"]
    seq_len    = data_config["keystroke_sequence_len"]
    target_len = hyperparams["target_len"]
    lr         = hyperparams["learning_rate"]
    embed_dim  = hyperparams["key_embedding_dim"]

    keystroke_data = KeystrokeData(training_data)
    vocab_size = keystroke_data.key_vocab_size()
    ranges     = keystroke_data.feature_ranges()

    epochs                  = int(sys.argv[1])
    metrics_every_n_epochs  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    check_val_every_n_epoch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
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
        compute_val_metrics=_make_metrics_fn(validation_data, target_len),
        compute_train_metrics=_make_metrics_fn(train_metrics_raw, target_len),
        metrics_every_n_epochs=metrics_every_n_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
