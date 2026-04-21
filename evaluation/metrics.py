import math
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

plt.style.use('seaborn-v0_8-bright')
plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams.update({"axes.grid": True, "grid.color": "black"})
mpl.rc('axes', edgecolor='black')
mpl.rcParams.update({'font.size': 13})


class Metric:

    # ── Core EER ──────────────────────────────────────────────────────────────

    @staticmethod
    def eer_compute(scores_g: torch.Tensor, scores_i: torch.Tensor, steps: int = 10_000):
        """Return (EER%, threshold) for genuine scores_g and impostor scores_i."""
        all_scores = torch.cat([scores_g, scores_i])
        lo, hi = all_scores.min().item(), all_scores.max().item()
        thresholds = torch.linspace(lo, hi, steps, device=all_scores.device)

        far = (scores_i.unsqueeze(1) >= thresholds).float().mean(0)
        frr = (scores_g.unsqueeze(1) <  thresholds).float().mean(0)

        idx = (far - frr).abs().argmin().item()
        eer = ((far[idx] + frr[idx]) / 2 * 100).item()
        return eer, thresholds[idx].item()

    # ── Distance helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _distance_humi(verify: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        """Mean L2 distance per verification session (HUMI layout)."""
        return torch.mean(torch.linalg.norm(verify - enroll.squeeze(1).unsqueeze(0), dim=-1), dim=-1)

    @staticmethod
    def _distance_hmog(verify: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        """Mean pairwise L2 distance per verification session (HMOG layout).
        verify: (V, Tv, D), enroll: (1, E, Te, D) — enroll has a leading dim from unsqueeze(0).
        Returns: (V,) mean distance per verification session.
        """
        enroll = enroll.squeeze(0)  # (E, Te, D)
        scores = []
        for ver_sess in verify:                           # ver_sess: (Tv, D)
            per_enroll = []
            for enr_sess in enroll:                       # enr_sess: (Te, D)
                d = torch.linalg.norm(
                    ver_sess.unsqueeze(1) - enr_sess.unsqueeze(0), dim=-1
                ).mean()                                  # scalar
                per_enroll.append(d)
            scores.append(torch.stack(per_enroll).mean())
        return torch.stack(scores)

    @staticmethod
    def _get_distance_fn(dataset: str):
        return Metric._distance_humi if dataset == "humi" else Metric._distance_hmog

    # ── Per-user EER helpers ──────────────────────────────────────────────────

    @staticmethod
    def _split_embeddings(embeddings: torch.Tensor, user: int, n_enroll: int):
        """Return (enroll, verify_genuine, verify_impostor) for one user."""
        enroll = embeddings[user, :n_enroll]
        genuine = embeddings[user, n_enroll:]
        impostor = torch.cat([
            embeddings[:user, n_enroll:].flatten(0, 1),
            embeddings[user+1:, n_enroll:].flatten(0, 1),
        ])
        return enroll.unsqueeze(0), genuine, impostor

    @staticmethod
    def _per_user_eer(embeddings: torch.Tensor, n_enroll: int, n_verify: int, dist_fn):
        eers, thresholds = [], []
        for i in range(len(embeddings)):
            enroll, genuine, impostor = Metric._split_embeddings(embeddings, i, n_enroll)
            all_ver = torch.cat([genuine, impostor])
            scores = dist_fn(all_ver, enroll)
            eer, thr = Metric.eer_compute(scores[:n_verify], scores[n_verify:])
            eers.append(eer)
            thresholds.append(thr)
        mean_eer = sum(eers) / len(eers) if eers else 0
        mean_thr = sum(thresholds) / len(thresholds) if thresholds else 0
        return 100 - mean_eer, mean_thr

    # ── Public EER methods ────────────────────────────────────────────────────

    @staticmethod
    def cal_user_eer_aalto(embeddings: torch.Tensor, n_enroll: int, n_verify: int):
        """EER using L2 norm on flat embeddings (Aalto dataset)."""
        eers, thresholds = [], []
        for i in range(len(embeddings)):
            enroll = embeddings[i, :n_enroll].unsqueeze(0)
            all_ver = torch.cat([
                embeddings[i, n_enroll:],
                embeddings[:i, n_enroll:].flatten(0, 1),
                embeddings[i+1:, n_enroll:].flatten(0, 1),
            ]).unsqueeze(1)
            scores = torch.mean(torch.linalg.norm(all_ver - enroll, dim=-1), dim=-1)
            eer, thr = Metric.eer_compute(scores[:n_verify], scores[n_verify:])
            eers.append(eer)
            thresholds.append(thr)
        mean_eer = sum(eers) / len(eers) if eers else 0
        mean_thr = sum(thresholds) / len(thresholds) if thresholds else 0
        return 100 - mean_eer, mean_thr

    @staticmethod
    def cal_user_eer(embeddings: torch.Tensor, n_enroll: int, n_verify: int, dataset: str):
        """EER using dataset-appropriate distance function."""
        return Metric._per_user_eer(embeddings, n_enroll, n_verify, Metric._get_distance_fn(dataset))

    # ── Usability / security metrics ──────────────────────────────────────────

    @staticmethod
    def _predictions(scores, threshold):
        return [1 if s <= threshold else 0 for s in scores]

    @staticmethod
    def calculate_usability(scores, threshold, periods, labels):
        """Fraction of legitimate-user time that is accepted."""
        preds = Metric._predictions(scores, threshold)
        total = accepted = 0
        for pred, period, label in zip(preds, periods, labels):
            if label == 0:
                continue
            total += period
            if pred == 1:
                accepted += period
        return accepted / total if total else 0

    @staticmethod
    def _window_lengths(preds, periods, labels, trigger_label, trigger_pred):
        """Generic helper: collect window lengths between trigger events."""
        values, time, active = [], 0, True
        for pred, period, label in zip(preds, periods, labels):
            if label == trigger_label:
                continue
            if pred == trigger_pred and active:
                values.append(time)
                time, active = 0, False
            elif pred != trigger_pred:
                time += period
                if not active:
                    active = True
        return values

    @staticmethod
    def calculate_TCR(scores, threshold, periods, labels):
        """Mean time (s) for impostor to be correctly rejected."""
        preds = Metric._predictions(scores, threshold)
        values = Metric._window_lengths(preds, periods, labels, trigger_label=1, trigger_pred=0)
        return np.mean(values) if values else 0

    @staticmethod
    def calculate_FRWI(scores, threshold, periods, labels):
        """Max window (min) during which a legitimate user is rejected."""
        preds = Metric._predictions(scores, threshold)
        values = Metric._window_lengths(preds, periods, labels, trigger_label=0, trigger_pred=1)
        return max(values) / 60 if values else 0

    @staticmethod
    def calculate_FAWI(scores, threshold, periods, labels):
        """Max window (min) during which an impostor is accepted."""
        preds = Metric._predictions(scores, threshold)
        values = Metric._window_lengths(preds, periods, labels, trigger_label=1, trigger_pred=0)
        return max(values) / 60 if values else 0

    # ── DET curve ─────────────────────────────────────────────────────────────

    @staticmethod
    def save_DET_curve(embeddings: torch.Tensor, n_enroll: int, n_verify: int,
                       dataset: str, results_path: str, steps: int = 10_000):
        dist_fn = Metric._get_distance_fn(dataset)
        n_users = len(embeddings)

        # Find global score range
        lo, hi = math.inf, -math.inf
        for i in range(n_users):
            enroll, genuine, impostor = Metric._split_embeddings(embeddings, i, n_enroll)
            scores = dist_fn(torch.cat([genuine, impostor]), enroll)
            lo = min(lo, scores.min().item())
            hi = max(hi, scores.max().item())

        thresholds = np.linspace(lo, hi, steps)
        far_frr_sum = np.zeros((steps, 2))

        for i in range(n_users):
            enroll, genuine, impostor = Metric._split_embeddings(embeddings, i, n_enroll)
            scores = dist_fn(torch.cat([genuine, impostor]), enroll).numpy()
            sg, si = scores[:n_verify], scores[n_verify:]
            far = (si[:, None] <= thresholds).mean(0) * 100
            frr = (sg[:, None] >  thresholds).mean(0) * 100
            far_frr_sum += np.stack([far, frr], axis=1)

        avg = far_frr_sum / n_users
        pd.DataFrame(avg, columns=["FAR", "FRR"]).to_csv(f"{results_path}/far-frr.csv", index=False)

    # ── t-SNE plot ────────────────────────────────────────────────────────────

    @staticmethod
    def save_PCA_curve(embeddings: torch.Tensor, session_count: int,
                       number_of_users: int, results_path: str):
        n_total = len(embeddings)
        users = np.random.choice(n_total, size=min(number_of_users, n_total), replace=False)

        flat = embeddings[users].mean(dim=-2).flatten(0, 1).cpu().numpy()
        coords = TSNE(n_iter=1000, perplexity=7).fit_transform(flat)
        labels = np.repeat(users, session_count)

        pd.DataFrame([[silhouette_score(coords, labels)]], columns=["Silhouette Score"]) \
          .to_csv(f"{results_path}/silhouette_score.csv", index=False)

        user_labels = [f"User {i+1}" for i, u in enumerate(users) for _ in range(session_count)]
        df = pd.DataFrame(coords, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
        df["Users"] = user_labels

        g = sns.relplot(data=df, x="t-SNE Dimension 1", y="t-SNE Dimension 2",
                        hue="Users", sizes=(10, 200))
        g.set(xscale="linear", yscale="linear")
        g.ax.xaxis.grid(True, "minor", linewidth=.25)
        g.ax.yaxis.grid(True, "minor", linewidth=.25)
        g.despine(left=True, bottom=True)
        plt.savefig(f'{results_path}/pca_graph.png', dpi=400)