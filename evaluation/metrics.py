from typing import Callable

import numpy as np
import torch

from utils.logger import get_logger

LOGGER = get_logger(__name__)
_N_THRESHOLDS = 500


class Metric:
    @staticmethod
    def scores_from_embeddings(
        embs: torch.Tensor,
        labels: torch.Tensor,
        max_impostor_pairs: int = 5_000_000,
        seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure pairwise window-level cosine similarities — matches SupCon semantics.

        Genuine: cosine sim of all (i, j), i < j, with same label.
        Impostor: random sample of pairs with different labels, capped at
        `max_impostor_pairs` to keep memory bounded.

        Embeddings are assumed L2-normalized.
        """
        embs   = embs.detach()
        labels = labels.detach()
        n = len(embs)

        # Sort by label so each user's embeddings are contiguous.
        order  = labels.argsort(stable=True)
        embs_s   = embs[order]
        labels_s = labels[order]
        users, counts = labels_s.unique_consecutive(return_counts=True)
        valid = counts >= 2
        users, counts = users[valid], counts[valid]
        splits = counts.cumsum(0)
        starts = torch.cat([torch.zeros(1, dtype=torch.long), splits[:-1]])

        genuine = []
        for s, c in zip(starts.tolist(), counts.tolist()):
            u = embs_s[s : s + c]
            sim = u @ u.T
            iu  = torch.triu_indices(c, c, offset=1)
            genuine.append(sim[iu[0], iu[1]])
        scores_g = torch.cat(genuine) if genuine else torch.zeros(0)

        # Impostor: sample random index pairs and keep those with different labels.
        gen = torch.Generator().manual_seed(seed)
        impostor = []
        collected = 0
        target = max_impostor_pairs
        while collected < target:
            k = min(target - collected, 1_000_000)
            i_idx = torch.randint(0, n, (k * 2,), generator=gen)
            j_idx = torch.randint(0, n, (k * 2,), generator=gen)
            keep  = (labels[i_idx] != labels[j_idx]) & (i_idx != j_idx)
            i_idx, j_idx = i_idx[keep][:k], j_idx[keep][:k]
            sim = (embs[i_idx] * embs[j_idx]).sum(dim=-1)
            impostor.append(sim)
            collected += len(sim)
            if len(sim) < k // 4:   # stop if too few different-label pairs left
                break
        scores_i = torch.cat(impostor) if impostor else torch.zeros(0)
        return scores_g, scores_i

    @staticmethod
    def eer_from_embeddings(embs: torch.Tensor, labels: torch.Tensor) -> float:
        """Pairwise window-level EER%. See scores_from_embeddings."""
        scores_g, scores_i = Metric.scores_from_embeddings(embs, labels)
        eer, _ = Metric.eer_compute(scores_g, scores_i)
        return eer

    @staticmethod
    def tar_at_far(
        scores_g: torch.Tensor,
        scores_i: torch.Tensor,
        far_targets: tuple[float, ...] = (0.01, 0.001, 0.0001),
    ) -> dict[float, float]:
        """TAR at each target FAR. Inputs are *similarities* (higher = more genuine).

        For each target FAR, the threshold is chosen so that exactly `far` of
        impostor scores exceed it. TAR = fraction of genuine scores ≥ threshold.
        """
        scores_g = scores_g.detach().cpu().float()
        scores_i = scores_i.detach().cpu().float()
        impostor_sorted = torch.sort(scores_i).values
        n_i = len(impostor_sorted)
        out: dict[float, float] = {}
        for far in far_targets:
            k = max(0, min(n_i - 1, n_i - int(round(far * n_i))))
            thr = impostor_sorted[k].item()
            tar = float((scores_g >= thr).float().mean().item())
            out[far] = tar
        return out

    @staticmethod
    def roc_auc(scores_g: torch.Tensor, scores_i: torch.Tensor) -> float:
        """ROC AUC via Mann-Whitney U. Inputs are *similarities* (higher = more genuine).

        AUC = P(genuine > impostor), with ties counted as 0.5.
        """
        scores_g = scores_g.detach().cpu().float()
        scores_i = scores_i.detach().cpu().float()
        all_scores = torch.cat([scores_g, scores_i])
        ranks = all_scores.argsort().argsort().float() + 1.0
        rank_sum_g = ranks[: len(scores_g)].sum().item()
        n_g, n_i = len(scores_g), len(scores_i)
        u = rank_sum_g - n_g * (n_g + 1) / 2
        return float(u / (n_g * n_i))

    @staticmethod
    def eer_compute(scores_g: torch.Tensor, scores_i: torch.Tensor, steps: int = 10_000):
        """Return (EER%, threshold) for genuine scores_g and impostor scores_i.

        Memory-efficient: uses sorted scores + searchsorted instead of an
        (n_scores × steps) boolean matrix.
        """
        scores_g = scores_g.detach().cpu().float()
        scores_i = scores_i.detach().cpu().float()
        all_scores = torch.cat([scores_g, scores_i])
        lo, hi = all_scores.min().item(), all_scores.max().item()
        thresholds = torch.linspace(lo, hi, steps)

        sorted_g = torch.sort(scores_g).values
        sorted_i = torch.sort(scores_i).values
        n_g, n_i = len(scores_g), len(scores_i)

        far = (n_i - torch.searchsorted(sorted_i, thresholds)).float() / n_i
        frr = torch.searchsorted(sorted_g, thresholds).float() / n_g

        idx = int((far - frr).abs().argmin().item())
        eer = ((far[idx] + frr[idx]) / 2 * 100).item()
        return eer, thresholds[idx].item()

    @staticmethod
    def evaluate_streams(
        score_fn: Callable[[dict], torch.Tensor],
        test_streams: list,
        single_streams: list,
        n_thresholds: int = _N_THRESHOLDS,
    ) -> dict:
        """
        Evaluate a CA scoring function on the test split.

        Window-boundary convention: a window starting at offset i covers events
        [i, i+L-1]. It "touches" the attack iff i + L - 1 >= t*, i.e. for any
        i >= boundary := max(0, t* - L + 1). Pre-attack windows are p_t[:boundary];
        post-attack windows are p_t[boundary:]. This matches the training label
        ("window contains any attack event ⇒ positive").

        Args:
            score_fn:       stream -> (n_windows,) tensor of p_t in [0, 1]
            test_streams:   list of attack + same_user stream dicts
            single_streams: list of single-session stream dicts (for Op.FPR)

        Returns dict with keys:
            ausc:      area under the usability-PTCR curve   (unit-less)
            ptcr:      fraction of attack streams detected   (per-stream)
            edd:       expected impostor keystrokes leaked   (keystrokes)
                       — averaged over ALL attacks; undetected → len(events) - t*.
            usability: fraction of pre-attack windows correctly silent
                       — per-window (≈ per-keystroke with stride-1, off-by-(L-1)
                       at the boundary).
            op_fpr:    macro-FPR over genuine streams: single sessions + the
                       suffix region of same-user streams (which is also fully
                       genuine but is not measured by Usability).
            tau_op:    operating threshold (where usability ≈ PTCR).
            Thresholds are sampled uniformly across the score range of
            (attack ∪ same_user) streams; single streams' scores can lie outside.
        """
        attack_streams    = [s for s in test_streams if s["stream_type"] == "attack"]
        same_user_streams = [s for s in test_streams if s["stream_type"] == "same_user"]
        all_test          = attack_streams + same_user_streams

        LOGGER.info(f"Scoring {len(attack_streams)} attack + {len(same_user_streams)} same-user streams")
        scored      = {id(s): score_fn(s) for s in all_test}
        LOGGER.info(f"Scoring {len(single_streams)} single-session streams")
        scored_sing = {id(s): score_fn(s) for s in single_streams}

        def _seq_len(s, p):
            return len(s["events"]) - len(p) + 1 if len(p) > 0 else None

        all_p = torch.cat([p for p in scored.values() if len(p) > 0])
        thresholds = torch.linspace(all_p.min().item(), all_p.max().item(), n_thresholds).tolist()

        usabilities, ptcrs = [], []
        for tau in thresholds:
            usab_vals = []
            for s in all_test:
                p = scored[id(s)]
                L = _seq_len(s, p)
                if L is None:
                    continue
                u = Metric._usability(p, s["t_star"], L, tau)
                if u is not None:
                    usab_vals.append(u)
            detected = 0
            for s in attack_streams:
                p = scored[id(s)]
                L = _seq_len(s, p)
                if L is None:
                    continue
                if Metric._detection_delay(p, s["t_star"], L, tau) is not None:
                    detected += 1
            usabilities.append(float(np.mean(usab_vals)) if usab_vals else 1.0)
            ptcrs.append(detected / max(len(attack_streams), 1))

        u_arr = np.array(usabilities)
        p_arr = np.array(ptcrs)

        order  = np.argsort(u_arr)
        ausc   = float(np.trapezoid(p_arr[order], u_arr[order]))
        op_idx = int(np.argmin(np.abs(u_arr - p_arr)))
        tau_op = thresholds[op_idx]

        # EDD = expected impostor keystrokes leaked before detection.
        # Window i covers events [i, i+L-1]. Detection at window w_start = boundary+d
        # ⇒ leaked = max(0, w_start + L - t*). Never detected ⇒ all impostor events.
        edd_vals = []
        for s in attack_streams:
            p_t = scored[id(s)]
            n_events = len(s["events"])
            L = _seq_len(s, p_t)
            if L is None:
                edd_vals.append(n_events - s["t_star"])
                continue
            boundary = max(0, s["t_star"] - L + 1)
            d = Metric._detection_delay(p_t, s["t_star"], L, tau_op)
            if d is not None:
                w_start = boundary + d
                edd_vals.append(max(0, w_start + L - s["t_star"]))
            else:
                edd_vals.append(n_events - s["t_star"])

        # Op.FPR: macro-FPR over genuine streams = singles + same-user suffix
        # (the part of same-user streams not measured by Usability).
        fpr_vals = [Metric._stream_fpr(scored_sing[id(s)], tau_op) for s in single_streams]
        for s in same_user_streams:
            p = scored[id(s)]
            L = _seq_len(s, p)
            if L is None:
                continue
            boundary = max(0, s["t_star"] - L + 1)
            suffix = p[boundary:]
            if len(suffix) > 0:
                fpr_vals.append(Metric._stream_fpr(suffix, tau_op))

        return {
            "ausc":      ausc,
            "ptcr":      ptcrs[op_idx],
            "edd":       float(np.mean(edd_vals)) if edd_vals else float("nan"),
            "usability": usabilities[op_idx],
            "op_fpr":    float(np.mean(fpr_vals)) if fpr_vals else float("nan"),
            "tau_op":    tau_op,
        }

    @staticmethod
    def print_results(results: dict, label: str = "Method") -> None:
        LOGGER.info(
            f"{label}  |  AUSC={results['ausc']:.4f}  PTCR={results['ptcr']:.4f}  "
            f"EDD={results['edd']:.1f}kstrokes  Usability={results['usability']:.4f}  "
            f"Op.FPR={results['op_fpr']:.4f}  τ_op={results['tau_op']:.4f}"
        )

    @staticmethod
    def _usability(p_t: torch.Tensor, t_star: int, seq_len: int, tau: float) -> float | None:
        """Pre-attack fraction correctly silent. Pre-attack = windows whose
        last event index < t* (window-start < max(0, t* - L + 1))."""
        if len(p_t) == 0:
            return None
        boundary = max(0, t_star - seq_len + 1)
        if boundary == 0:
            return None
        pre = p_t[:boundary]
        return float((pre < tau).float().mean().item()) if len(pre) > 0 else None

    @staticmethod
    def _detection_delay(p_t: torch.Tensor, t_star: int, seq_len: int, tau: float) -> int | None:
        """Offset (in window indices) from the first window that touches the
        attack (w_start = max(0, t* - L + 1)) to the first window where p_t >= tau."""
        boundary = max(0, t_star - seq_len + 1)
        post = p_t[boundary:]
        hits = (post >= tau).nonzero(as_tuple=False)
        return int(hits[0].item()) if len(hits) > 0 else None

    @staticmethod
    def _stream_fpr(p_t: torch.Tensor, tau: float) -> float:
        return float((p_t >= tau).float().mean().item()) if len(p_t) > 0 else 0.0
