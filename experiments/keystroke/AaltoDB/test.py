import math
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[3] / "utils"))
sys.path.append(str(Path(__file__).resolve().parents[3] / "evaluation"))
from Config import Config
from metrics import Metric
from model import KeystrokeTransformer

# ── Plot defaults ────────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-bright")
plt.rcParams["axes.facecolor"] = "white"
mpl.rcParams.update({"axes.grid": True, "grid.color": "black", "font.size": 13})
mpl.rc("axes", edgecolor="black")

# ── Paths ────────────────────────────────────────────────────────────────────

HERE          = Path(__file__).resolve().parent
RESULTS_PATH  = HERE / "results"
MODELS_PATH   = HERE / "best_models"
AALTO_PREP    = Path("data/AaltoDB/prep_data")


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess():
    data = pd.read_csv(AALTO_PREP / "keystroke_data.csv")
    assert not data.isnull().values.any(), "Dataset contains NaN values"

    grouped = {
        user: [g[["press_time", "release_time", "key_code"]].to_numpy()
               for _, g in sessions.groupby("session_id")]
        for user, sessions in data.groupby("user_id")
    }

    # Keep only users with exactly 15 sessions
    grouped = {u: s for u, s in grouped.items() if len(s) == 15}
    sessions = list(grouped.values())
    print(f"Users after filtering: {len(sessions)}")

    training_data   = sessions[:-1050]
    validation_data = sessions[-1050:-1000]
    testing_data    = sessions[-1000:]

    for name, split in [("training", training_data),
                        ("validation", validation_data),
                        ("testing", testing_data)]:
        with open(AALTO_PREP / f"{name}_data.pickle", "wb") as f:
            pickle.dump(split, f)


# ── Dataset ──────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    SESSIONS_PER_USER = 15

    def __init__(self, eval_data, seq_len):
        self.eval_data = eval_data
        self.seq_len   = seq_len

    def __len__(self):
        return math.ceil(len(self.eval_data) * self.SESSIONS_PER_USER)

    def __getitem__(self, idx):
        sequence = self.eval_data[idx // self.SESSIONS_PER_USER][idx % self.SESSIONS_PER_USER]
        return self._pad(sequence)

    def _pad(self, sequence):
        n = len(sequence)
        if n == self.seq_len:
            return sequence
        if n < self.seq_len:
            padding = np.zeros((self.seq_len - n, 10))
            return np.append(sequence, padding, axis=0)
        return sequence[:self.seq_len]


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_normalize_features(dataset):
    """Vectorised: no Python loop over rows."""
    for user_sequences in dataset:
        for idx, seq in enumerate(user_sequences):
            n = len(seq)
            press   = seq[:, 0]
            release = seq[:, 1]
            keys    = seq[:, 2]

            # Shifted arrays — pad last 1 or 2 rows with 0
            p1 = np.concatenate([press[1:],   [0]])
            r1 = np.concatenate([release[1:], [0]])
            p2 = np.concatenate([press[2:],   [0, 0]])
            r2 = np.concatenate([release[2:], [0, 0]])

            features = np.stack([
                (release - press),          # hold latency
                (p1 - release),             # di_ud
                (p1 - press),               # di_dd
                (r1 - release),             # di_uu
                (r1 - press),               # di_du
                (p2 - release),             # tri_ud
                (p2 - press),               # tri_dd
                (r2 - release),             # tri_uu
                (r2 - press),               # tri_du
                keys,                       # key_code (normalised below)
            ], axis=1).astype(np.float64)

            features[:, :9] /= 1000        # timing → seconds
            features[:, 9]  /= 255         # key code → [0,1]

            # Zero out invalid tail entries
            features[-1, 1:5]  = 0         # no next key for last row
            features[-2:, 5:9] = 0         # no key+2 for last two rows

            user_sequences[idx] = features


# ── Score helpers ────────────────────────────────────────────────────────────

def _scores_all(feature_embeddings, num_enroll):
    """
    Compute scores for every user at once with a single batched norm call.
    Returns a list of score tensors, one per user.
    """
    n_users, n_sessions, emb_dim = feature_embeddings.shape
    n_verify = n_sessions - num_enroll

    enroll = feature_embeddings[:, :num_enroll]           # (U, E, D)
    verify = feature_embeddings[:, num_enroll:]           # (U, V, D)

    all_scores = []
    for i in range(n_users):
        enroll_i = enroll[i].unsqueeze(0)                 # (1, E, D)

        impostor_seqs = torch.cat([
            verify[:i].flatten(0, 1),
            verify[i+1:].flatten(0, 1),
        ], dim=0)                                         # ((U-1)*V, D)

        genuine_seqs  = verify[i]                         # (V, D)
        all_seqs = torch.cat([genuine_seqs, impostor_seqs], dim=0).unsqueeze(1)  # (N,1,D)

        scores = torch.mean(torch.linalg.norm(all_seqs - enroll_i, dim=-1), dim=-1)
        all_scores.append(scores)

    return all_scores


# ── Evaluation ───────────────────────────────────────────────────────────────

def get_evaluate_results(feature_embeddings, num_enroll_sessions):
    n_verify   = 15 - num_enroll_sessions
    all_scores = _scores_all(feature_embeddings, num_enroll_sessions)
    n_users    = feature_embeddings.shape[0]
    metrics    = {"acc": [], "usability": [], "tcr": [], "fawi": [], "frwi": []}

    for i, scores in enumerate(all_scores):
        labels  = torch.tensor([1] * n_verify + [0] * (n_users - 1))
        periods = get_periods(i)
        acc, threshold = Metric.eer_compute(scores[:n_verify], scores[n_verify:])
        metrics["acc"].append(acc)
        metrics["usability"].append(Metric.calculate_usability(scores, threshold, periods, labels))
        metrics["tcr"].append(Metric.calculate_TCR(scores, threshold, periods, labels))
        metrics["fawi"].append(Metric.calculate_FAWI(scores, threshold, periods, labels))
        metrics["frwi"].append(Metric.calculate_FRWI(scores, threshold, periods, labels))

    return (
        100 - np.mean(metrics["acc"]),
        np.mean(metrics["usability"]),
        np.mean(metrics["tcr"]),
        np.mean(metrics["fawi"]),
        np.mean(metrics["frwi"]),
    )


def get_periods(user_id):
    def window_time(seq):
        return np.sum(seq, axis=0)[2] + seq[-1][0]

    periods = [window_time(testing_data[user_id][10 + j]) for j in range(5)]
    periods += [window_time(testing_data[i][10]) for i in range(len(testing_data)) if i != user_id]
    return periods


# ── DET curve ────────────────────────────────────────────────────────────────

def _threshold_range(ini, fin, steps=10_000):
    paso = (fin - ini) / steps
    t = ini - paso
    while t < fin + paso:
        yield t
        t += paso

def _score_range(feature_embeddings, num_enroll):
    all_scores = []
    for i in range(feature_embeddings.shape[0]):
        scores = _scores_all(feature_embeddings, num_enroll)
        all_scores.append(scores)
    return all_scores

def save_DET_curve(feature_embeddings, num_enroll_sessions):
    n_verify   = 15 - num_enroll_sessions
    all_scores = _scores_all(feature_embeddings, num_enroll_sessions)

    combined = torch.cat(all_scores)
    _min, _max = combined.min().item(), combined.max().item()

    eer_positions = [
        _eer_index(s[:n_verify], s[n_verify:], _min, _max)
        for s in all_scores
    ]
    fix_eer_pos   = max(eer_positions)
    max_ele_count = fix_eer_pos - min(eer_positions)

    values = div = 0
    for i, scores in enumerate(all_scores):
        far_frrs = _get_far_frr(scores[:n_verify], scores[n_verify:],
                                _min, _max, fix_eer_pos, max_ele_count, eer_positions[i])
        if isinstance(values, int):
            values, div = far_frrs
        else:
            values += far_frrs[0]
            div    += far_frrs[1]

    pd.DataFrame(values / div, columns=["FAR", "FRR"]).to_csv(RESULTS_PATH / "far-frr.csv")


def _vectorised_far_frr(scores_g, scores_i, ini, fin, steps=10_000):
    """
    Compute FAR and FRR for all thresholds in one shot using broadcasting.
    Returns (far, frr) as np arrays of shape (steps+2,).
    """
    paso       = (fin - ini) / steps
    thresholds = torch.arange(ini - paso, fin + 2 * paso, paso)   # (T,)

    # Broadcasting: (T,1) vs (N,) → (T, N) → mean per threshold
    far = (scores_i.unsqueeze(0) >= thresholds.unsqueeze(1)).float().mean(dim=1)
    frr = (scores_g.unsqueeze(0) <  thresholds.unsqueeze(1)).float().mean(dim=1)

    return far.cpu().numpy(), frr.cpu().numpy()

def _get_far_frr(scores_g, scores_i, ini, fin, fix_eer_pos, max_ele_count, eer_pos):
    far, frr     = _vectorised_far_frr(scores_g, scores_i, ini, fin)
    far_frr      = np.stack([far, frr], axis=1)

    adding_count = fix_eer_pos - eer_pos
    if adding_count:
        front_pad = np.zeros((adding_count, 2))
        far_frr   = np.vstack([front_pad, far_frr[:-adding_count]])
    else:
        front_pad = np.zeros((0, 2))

    back_pad = np.ones((len(far_frr) - adding_count, 2))
    return far_frr, np.vstack([front_pad, back_pad])

def _eer_index(scores_g, scores_i, ini, fin):
    far, frr = _vectorised_far_frr(scores_g, scores_i, ini, fin)
    gap = np.abs(far - frr)
    return int(np.argmin(gap))


def _get_far_frr(scores_g, scores_i, ini, fin, fix_eer_pos, max_ele_count, eer_pos):
    far_frr = [
        [torch.count_nonzero(scores_i >= t).item() / len(scores_i),
         torch.count_nonzero(scores_g <  t).item() / len(scores_g)]
        for t in _threshold_range(ini, fin)
    ]
    adding_count = fix_eer_pos - eer_pos
    front_pad    = [[0.0, 0.0]] * adding_count
    if adding_count:
        far_frr = front_pad + far_frr[:-adding_count]
    back_pad = [[1.0, 1.0]] * (len(far_frr) - adding_count)
    return np.array(far_frr), np.array(front_pad + back_pad)


# ── t-SNE / PCA plot ─────────────────────────────────────────────────────────

def save_PCA_curve(feature_embeddings, num_enroll_sessions, number_of_users):
    # Sample 10 unique random users
    users = list({np.random.randint(0, 1000) for _ in range(1000)})[:10]

    flat  = feature_embeddings[users].flatten(start_dim=0, end_dim=1).cpu().numpy()
    tsne  = TSNE(n_iter=1000, perplexity=14).fit_transform(flat)
    labels = [u for u in users for _ in range(15)]

    pd.DataFrame([[silhouette_score(tsne, labels)]], columns=["Silhouette Score"]) \
      .to_csv(RESULTS_PATH / "silhouette_score.csv")

    df = pd.DataFrame(tsne, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
    df["Users"] = [f"User {i+1}" for i in range(number_of_users) for _ in range(15)]

    COLORS = ["red","green","blue","black","blueviolet",
              "orange","grey","brown","deeppink","purple"]
    g = sns.relplot(data=df, x="t-SNE Dimension 1", y="t-SNE Dimension 2",
                    hue="Users", sizes=(10, 200), palette=COLORS)
    g.set(xscale="linear", yscale="linear")
    g.ax.xaxis.grid(True, "minor", linewidth=0.25)
    g.ax.yaxis.grid(True, "minor", linewidth=0.25)
    g.despine(left=True, bottom=True)
    g._legend.remove()
    plt.savefig(RESULTS_PATH / "pca_graph.png", dpi=400)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config     = Config()
    cfg        = config.get_config_dict()
    use_gpu    = cfg["GPU"] == "True"
    device     = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    metric_arg = sys.argv[1]
    model_file = sys.argv[2]

    subprocess.run(f"gdown {cfg['preprocessed_data']['aalto']['keystroke']}", shell=True)

    preprocess()
    RESULTS_PATH.mkdir(exist_ok=True)

    with open("data/AaltoDB/prep_data/testing_data.pickle", "rb") as f:
        testing_data = pickle.load(f)
    extract_normalize_features(testing_data)

    seq_len    = cfg["data"]["keystroke_sequence_len"]
    batch_size = cfg["hyperparams"]["batch_size"]["aalto"]

    model_path = MODELS_PATH / model_file
    test_model = torch.load(model_path, map_location=device)
    test_model.to(device)
    test_model.eval()

    dataloader = DataLoader(TestDataset(testing_data, seq_len), batch_size=batch_size)
    with torch.no_grad():
        feature_embeddings = torch.cat(
            [test_model(batch.float().to(device)) for batch in dataloader], dim=0
        ).view(len(testing_data), 15, 64)

    if metric_arg == "basic":
        res = get_evaluate_results(feature_embeddings, 10)
        pd.DataFrame([list(res)], columns=["eer", "usability", "tcr", "fawi", "frwi"]) \
          .to_csv(RESULTS_PATH / "basic.csv")
    elif metric_arg == "det":
        save_DET_curve(feature_embeddings, 10)
    elif metric_arg == "pca":
        save_PCA_curve(feature_embeddings, 10, 10)
