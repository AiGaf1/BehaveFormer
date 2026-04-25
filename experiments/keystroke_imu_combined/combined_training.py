"""Shared training infrastructure for all keystroke_imu_combined experiments."""
import math
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.common.loss import TripletLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

from Config import Config  # noqa: E402
from metrics import Metric  # noqa: E402

TRAINING_PICKLE   = "training_keystroke_imu_data_all.pickle"
VALIDATION_PICKLE = "validation_keystroke_imu_data_all.pickle"
TESTING_PICKLE    = "testing_keystroke_imu_data_all.pickle"

HMOG_IMU_SCALE_MAP = {
    0: 10, 1: 10, 2: 10,
    3: 1000, 4: 1000, 5: 1000, 15: 1000, 16: 1000, 17: 1000,
    24: 100, 25: 100, 26: 100,
    27: 10000, 28: 10000, 29: 10000,
}

HUMI_IMU_SCALE_MAP = {
    0: 10, 1: 10, 2: 10,
    3: 1000, 4: 1000, 5: 1000,
    15: 100, 16: 100, 17: 100,
    24: 1000, 25: 1000, 26: 1000,
    27: 1000, 28: 1000, 29: 1000,
}


def _scale_imu_features(data, imu_scale_map):
    for user in data:
        for session in user:
            for i, sequence in enumerate(session):
                imu = sequence[1].astype(np.float64, copy=True)
                for col, divisor in imu_scale_map.items():
                    imu[:, col] /= divisor
                session[i][1] = imu


@dataclass(frozen=True)
class CombinedSpec:
    dataset_key: str           # "hmog" or "humi"
    imu_feature_count_key: str # "one_type", "two_types", or "all"
    imu_columns: object        # numpy index, e.g. slice(None,12) or [0..11]+[24..35]


def _pickle_path(dataset_dir_name, name):
    prep = PROJECT_ROOT / "data" / dataset_dir_name / "prep_data" / name
    root = PROJECT_ROOT / name
    return prep if prep.exists() or not root.exists() else root


def _maybe_download(file_id, path):
    if path.exists() or not file_id:
        return
    subprocess.run(f"gdown {file_id}", shell=True, check=True, cwd=path.parent)


class _TripletDataset(torch.utils.data.Dataset):
    def __init__(self, data, batch_size, epoch_batch_count, imu_columns):
        self.data = data
        self.length = batch_size * epoch_batch_count
        self.imu_columns = imu_columns

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        n = len(self.data)
        g = np.random.randint(n)
        imp = np.random.randint(n - 1)
        if imp >= g:
            imp += 1
        s1, s2 = np.random.choice(len(self.data[g]), size=2, replace=False)
        si = np.random.randint(len(self.data[imp]))
        a = self.data[g][s1][np.random.randint(len(self.data[g][s1]))]
        p = self.data[g][s2][np.random.randint(len(self.data[g][s2]))]
        ne = self.data[imp][si][np.random.randint(len(self.data[imp][si]))]
        return self._select(a), self._select(p), self._select(ne)

    def _select(self, seq):
        return [seq[0], seq[1][:, self.imu_columns]]


class _EvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, imu_columns):
        self.data = data
        self.n_sessions = len(data[0])
        self.n_seqs = len(data[0][0])
        self.imu_columns = imu_columns

    def __len__(self):
        return len(self.data) * self.n_sessions * self.n_seqs

    def __getitem__(self, idx):
        user    = idx // (self.n_sessions * self.n_seqs)
        session = (idx // self.n_seqs) % self.n_sessions
        seq     = idx % self.n_seqs
        s = self.data[user][session][seq]
        return [s[0], s[1][:, self.imu_columns]]


def _collate(batch):
    anchors, positives, negatives = zip(*batch)
    def _stack(items):
        return [torch.stack([torch.as_tensor(x[0], dtype=torch.float32) for x in items]),
                torch.stack([torch.as_tensor(x[1], dtype=torch.float32) for x in items])]
    return _stack(anchors), _stack(positives), _stack(negatives)


def _collate_eval(batch):
    return [torch.stack([torch.as_tensor(x[0], dtype=torch.float32) for x in batch]),
            torch.stack([torch.as_tensor(x[1], dtype=torch.float32) for x in batch])]


@torch.no_grad()
def _evaluate(model, data, batch_size, trg_len, n_enroll, n_verify, dataset_key, imu_columns, device):
    model.eval()
    dataset = _EvalDataset(data, imu_columns)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_eval)
    embeddings = torch.cat([model([b[0].to(device), b[1].to(device)]) for b in loader])
    n_users, n_sessions, n_seqs = len(data), dataset.n_sessions, dataset.n_seqs
    return Metric.cal_user_eer(embeddings.view(n_users, n_sessions, n_seqs, trg_len), n_enroll, n_verify, dataset_key)[0]


def run_combined_training(*, spec: CombinedSpec, dataset_dir_name: str, model_factory, here: Path, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("Epoch count required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config().get_config_dict()
    hp     = config["hyperparams"]
    data_cfg = config["data"]
    ids    = config["preprocessed_data"][spec.dataset_key]

    train_path = _pickle_path(dataset_dir_name, TRAINING_PICKLE)
    val_path   = _pickle_path(dataset_dir_name, VALIDATION_PICKLE)
    _maybe_download(ids["train"], train_path)
    _maybe_download(ids["val"],   val_path)
    _maybe_download(ids.get("test"), _pickle_path(dataset_dir_name, TESTING_PICKLE))

    with open(train_path, "rb") as f:
        training_data = pickle.load(f)
    with open(val_path, "rb") as f:
        validation_data = pickle.load(f)

    imu_map = HMOG_IMU_SCALE_MAP       if spec.dataset_key == "hmog" else HUMI_IMU_SCALE_MAP
    val_limit = 50 if spec.dataset_key == "hmog" else 1
    for user in validation_data:
        for idx, session in enumerate(user):
            user[idx] = session[:val_limit]

    _scale_imu_features(training_data,   imu_map)
    _scale_imu_features(validation_data, imu_map)

    batch_size        = hp["batch_size"][spec.dataset_key]
    epoch_batch_count = hp["epoch_batch_count"][spec.dataset_key]
    trg_len           = hp["target_len"]
    n_enroll          = hp["number_of_enrollment_sessions"][spec.dataset_key]
    n_verify          = hp["number_of_verify_sessions"][spec.dataset_key]
    ks_features       = hp["keystroke_feature_count"][spec.dataset_key]
    imu_features      = hp["imu_feature_count"][spec.imu_feature_count_key]
    ks_len            = data_cfg["keystroke_sequence_len"]
    imu_len           = data_cfg["imu_sequence_len"]

    model = model_factory(ks_features, imu_features, ks_len, imu_len, trg_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    loss_fn   = TripletLoss()

    best_dir = here / "best_models"
    ckpt_dir = here / "checkpoints"
    best_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    total_epochs = int(argv[0])
    start_epoch  = 0
    best_eer     = math.inf

    if len(argv) > 1:
        ckpt = torch.load(ckpt_dir / f"training_{argv[1]}.tar", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt["epoch"]
        best_eer     = ckpt["eer"]
        total_epochs += start_epoch

    dataset    = _TripletDataset(training_data, batch_size, epoch_batch_count, spec.imu_columns)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate, pin_memory=True, num_workers=4)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for anchor, positive, negative in dataloader:
            a = model([anchor[0].to(device),   anchor[1].to(device)])
            p = model([positive[0].to(device),  positive[1].to(device)])
            n = model([negative[0].to(device),  negative[1].to(device)])
            loss = loss_fn(a, p, n)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(dataloader)

        eer = _evaluate(model, validation_data, batch_size, trg_len, n_enroll, n_verify,
                        spec.dataset_key, spec.imu_columns, device)
        print(f"Epoch {epoch+1:>4d} | loss {total_loss:.6f} | EER {eer:.6f} | {time.time()-t0:.1f}s")

        if eer < best_eer:
            print(f"  EER improved {best_eer:.6f} -> {eer:.6f}")
            best_eer = eer
            torch.save(model, best_dir / f"epoch_{epoch+1}_eer_{eer}.pt")

        if (epoch + 1) % 50 == 0:
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "eer": best_eer},
                       ckpt_dir / f"training_{epoch+1}.tar")

    torch.save({"epoch": total_epochs, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "eer": best_eer},
               ckpt_dir / f"training_{total_epochs}.tar")
