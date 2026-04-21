import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import Model
sys.path.append(str((Path(__file__)/"../../../../../utils").resolve()))
sys.path.append(str((Path(__file__)/"../../../../../evaluation").resolve()))
from Config import Config
from metrics import Metric

# ── Constants ────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = Path(__file__).parent


# ── Data helpers ──────────────────────────────────────────────────────────────

def scale(data):
    """Normalise keystroke and IMU arrays in-place."""
    for user in data:
        for session in user:
            for i, (ks, imu) in enumerate(session):
                ks = np.array(ks, dtype=np.float32)
                imu = np.array(imu, dtype=np.float32)

                ks[:, :9]  /= 1_000
                ks[:, 9]   /= 255

                imu[:, [0, 1, 2]]            /= 10
                imu[:, [3, 4, 5, 15, 16, 17]] /= 1_000
                imu[:, [24, 25, 26]]          /= 100
                imu[:, [27, 28, 29]]          /= 10_000

                np.nan_to_num(ks,  copy=False)
                np.nan_to_num(imu, copy=False)
                session[i] = [ks, imu]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Datasets ──────────────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    """Yields random (anchor, positive, negative) triplets."""

    def __init__(self, data, batch_size, epoch_batch_count):
        self.data = data
        self.length = batch_size * epoch_batch_count
        self.n_users = len(data)
        self.n_sessions = len(data[0])

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = np.random.randint  # shorthand

        g = rng(0, self.n_users)
        imp = rng(0, self.n_users - 1)
        if imp >= g:
            imp += 1  # guaranteed different, no loop needed

        s1, s2 = np.random.choice(self.n_sessions, size=2, replace=False)
        si = rng(0, self.n_sessions)

        anchor   = self.data[g][s1][rng(0, len(self.data[g][s1]))]
        positive = self.data[g][s2][rng(0, len(self.data[g][s2]))]
        negative = self.data[imp][si][rng(0, len(self.data[imp][si]))]

        return anchor, positive, negative


class TestDataset(Dataset):
    """Flattens eval data into (user × session × sequence) items."""

    def __init__(self, data):
        if len(data) < 2:
            raise ValueError("Evaluation needs at least 2 users.")
        self.data = data
        self.n_sessions = len(data[0])
        self.n_seqs = min(len(s) for u in data for s in u)
        if self.n_sessions == 0 or self.n_seqs == 0:
            raise ValueError("Evaluation data must have at least one session and sequence.")

    def __len__(self):
        return len(self.data) * self.n_sessions * self.n_seqs

    def __getitem__(self, idx):
        user    = idx // (self.n_sessions * self.n_seqs)
        session = (idx // self.n_seqs) % self.n_sessions
        seq     = idx % self.n_seqs
        return self.data[user][session][seq]


# ── Loss ──────────────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = (anchor - positive).pow(2).sum(dim=1).sqrt()
        d_neg = (anchor - negative).pow(2).sum(dim=1).sqrt()
        return torch.relu(d_pos - d_neg + self.margin).mean()


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, batch_size, trg_len, n_enroll, n_verify):
    model.eval()
    loader = DataLoader(TestDataset(data), batch_size=batch_size)

    embeddings = torch.cat(
        [model([b[0].float().to(DEVICE), b[1].float().to(DEVICE)]) for b in loader]
    )

    n_users, n_sessions, n_seqs = len(data), loader.dataset.n_sessions, loader.dataset.n_seqs
    embeddings = embeddings.view(n_users, n_sessions, n_seqs, trg_len)

    n_verify = min(n_verify, n_sessions - n_enroll)
    return Metric.cal_user_eer(embeddings, n_enroll, n_verify, "hmog")[0]


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for anchor, positive, negative in dataloader:
        a = model([anchor[0].float().to(DEVICE),   anchor[1].float().to(DEVICE)])
        p = model([positive[0].float().to(DEVICE), positive[1].float().to(DEVICE)])
        n = model([negative[0].float().to(DEVICE), negative[1].float().to(DEVICE)])

        loss = loss_fn(a, p, n)
        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    config = Config()
    cfg = config.get_config_dict()
    data_cfg = cfg["data"]
    hp = cfg["hyperparams"]
    ids = cfg["preprocessed_data"]["hmog"]
 
    data_files = [
        (ids["train"], Path("data/HMOGDB/prep_data/training_keystroke_imu_data_all.pickle")),
        (ids["test"],  Path("data/HMOGDB/prep_data/testing_keystroke_imu_data_all.pickle")),
        (ids["val"],   Path("data/HMOGDB/prep_data/validation_keystroke_imu_data_all.pickle")),
    ]
    for gid, path in data_files:
        if gid and not path.exists():
            subprocess.run(f"gdown {gid}", shell=True, check=True)
 
    training_data   = load_pickle("data/HMOGDB/prep_data/training_keystroke_imu_data_all.pickle")
    validation_data = load_pickle("data/HMOGDB/prep_data/validation_keystroke_imu_data_all.pickle")
 
    # Limit validation sequences
    for user in validation_data:
        for idx, session in enumerate(user):
            user[idx] = session[:50]
 
    scale(training_data)
    scale(validation_data)
 
    batch_size      = hp["batch_size"]["hmog"]
    trg_len         = hp["target_len"]
    n_enroll        = hp["number_of_enrollment_sessions"]["hmog"]
    n_verify        = hp["number_of_verify_sessions"]["hmog"]
 
    best_dir  = HERE / "best_models"
    ckpt_dir  = HERE / "checkpoints"
    best_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
 
    model = Model(
        hp["keystroke_feature_count"]["hmog"],
        hp["imu_feature_count"]["all"],
        data_cfg["keystroke_sequence_len"],
        data_cfg["imu_sequence_len"],
        trg_len,
    ).to(DEVICE)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    loss_fn   = TripletLoss()
 
    total_epochs = int(sys.argv[1])
    start_epoch  = 0
    best_eer     = math.inf
 
    if len(sys.argv) > 2:
        ckpt = torch.load(ckpt_dir / f"training_{sys.argv[2]}.tar", map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt["epoch"]
        best_eer     = ckpt["eer"]
        total_epochs += start_epoch
 
    dataset    = TrainDataset(training_data, batch_size, hp["epoch_batch_count"]["hmog"])
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
 
    for epoch in range(start_epoch, total_epochs):
        t0   = time.time()
        loss = train(model, dataloader, optimizer, loss_fn)
        eer  = evaluate(model, validation_data, batch_size, trg_len, n_enroll, n_verify)
        print(f"Epoch {epoch+1:>4d} | loss {loss:.6f} | EER {eer:.6f} | {time.time()-t0:.1f}s")
 
        if eer < best_eer:
            print(f"  ↳ EER improved {best_eer:.6f} → {eer:.6f}")
            best_eer = eer
            torch.save(model, best_dir / f"epoch_{epoch+1}_eer_{eer}.pt")
 
        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "eer": best_eer,
            }, ckpt_dir / f"training_{epoch+1}.tar")
 
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eer": best_eer,
    }, ckpt_dir / f"training_{total_epochs}.tar")
 
 
if __name__ == "__main__":
    main()