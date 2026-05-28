"""
Stage 1 encoder evaluation — window-level verification + Stage-1-only CA ablation.

Reports:
  - Verification: EER, AUC, TAR @ FAR ∈ {1%, 0.1%, 0.01%}, score histogram
  - CA ablation:  AUSC, PTCR, EDD, Usability, Op.FPR, τ_op (cosine to enrollment
    template — no Stage 2 detector). This row goes in the paper's ablation table
    against Stage 2.

Usage:
    cd <project_root>
    python -m experiments.verification.test [checkpoint.ckpt]
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.windows import PureWindowDataset  # noqa: E402
from data.AaltoDB.streams import ENROLL_END, load_streams  # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from evaluation.ca_baseline import build_user_template, score_stream_baseline  # noqa: E402
from evaluation.verification import behaveformer_eer, encode_windows  # noqa: E402
from experiments.verification.model import Encoder  # noqa: E402
from experiments.verification.train import (  # noqa: E402
    DROPOUT,
    HEADS,
    KEY_EMB,
    LFF_FEATURES,
    NUM_LAYERS,
    SEQ_LEN,
)

SAMPLES_PER_USER = 8   # legacy: pairs_per_user for PureWindowDataset pairwise EER
from utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

_BEST_MODELS_DIR = Path(__file__).resolve().parent / "best_models"
_EER_RX = re.compile(r"eer_(\d+\.\d+)")


def _pick_best_ckpt() -> Path:
    ckpts = list(_BEST_MODELS_DIR.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {_BEST_MODELS_DIR}")
    parsed = [(float(m.group(1)), c) for c in ckpts if (m := _EER_RX.search(c.name))]
    if parsed:
        return min(parsed, key=lambda x: x[0])[1]
    return sorted(ckpts)[-1]


def test(ckpt_path: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path or _pick_best_ckpt()
    LOGGER.info(f"Loading checkpoint: {ckpt_path.name}")

    train_data, _, test_data = load_aalto()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = Encoder(
        seq_len=SEQ_LEN, vocab_size=aalto_vocab_size(train_data),
        key_emb=KEY_EMB, feature_ranges=aalto_feature_ranges(train_data),
        lff_features=LFF_FEATURES, num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT,
    ).to(device)
    state = ckpt["state_dict"]
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items()
                             if k.startswith("encoder.")})
    encoder.eval()

    ds = PureWindowDataset(test_data, seq_len=SEQ_LEN,
                           users_per_epoch=len(test_data),
                           pairs_per_user=SAMPLES_PER_USER,
                           max_users=len(test_data))
    ds.resample(seed=0)

    embs = encode_windows(encoder, ds, device)
    scores_g, scores_i = Metric.scores_from_embeddings(embs, ds.labels)
    eer, thr = Metric.eer_compute(scores_g, scores_i)
    auc      = Metric.roc_auc(scores_g, scores_i)
    tar_far  = Metric.tar_at_far(scores_g, scores_i)

    LOGGER.info("Verification (test set, pairwise window-level)")
    LOGGER.info(f"  N users={len(test_data):,}  N windows={len(ds):,}  "
                f"(M={SAMPLES_PER_USER})  genuine_pairs={len(scores_g):,}  "
                f"impostor_pairs={len(scores_i):,}")
    LOGGER.info(f"  EER={eer:.2f}%  threshold={thr:.4f}  AUC={auc:.4f}")
    tar_str = "  ".join(f"FAR={far*100:.2f}%→TAR={tar*100:.2f}%" for far, tar in tar_far.items())
    LOGGER.info(f"  {tar_str}")

    # ── BehaveFormer protocol (IJCB 2023) verification — directly comparable ──
    LOGGER.info("Verification — BehaveFormer protocol (5-session enrollment, mean Euclidean)")
    for n_users in (1000, len(test_data)):
        bf = behaveformer_eer(encoder, test_data, SEQ_LEN, device,
                              n_enroll=5, max_users=n_users, seed=0)
        LOGGER.info(f"  N_test_users={bf['n_users']:,}  E={bf['n_enroll']}  "
                    f"genuine={bf['n_genuine']:,}  impostor={bf['n_impostor']:,}")
        LOGGER.info(f"  EER={bf['eer']:.2f}%  AUC={bf['auc']:.4f}  "
                    f"thr_dist={bf['threshold_dist']:.4f}")
        tar_str_bf = "  ".join(f"FAR={far*100:.2f}%→TAR={tar*100:.2f}%"
                               for far, tar in bf["tar_at_far"].items())
        LOGGER.info(f"  {tar_str_bf}")
        if n_users >= len(test_data):
            break  # avoid re-running when test split < 1000

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores_g.cpu().numpy(), bins=80, alpha=0.6, label="genuine",  density=True, color="C0")
    ax.hist(scores_i.cpu().numpy(), bins=80, alpha=0.6, label="impostor", density=True, color="C3")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"EER threshold ({thr:.3f})")
    ax.set_xlabel("Cosine similarity between window pairs")
    ax.set_ylabel("Density")
    ax.set_title(f"Stage 1 pairwise score distribution — {ckpt_path.stem}")
    ax.legend()
    hist_path = ckpt_path.with_name(ckpt_path.stem + "_scores.png")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    LOGGER.info(f"Histogram saved to {hist_path.name}")

    # ── CA ablation: Stage 1 only (cosine to enrollment template) ─────────────
    LOGGER.info("Building enrollment templates for CA ablation …")
    rng = np.random.default_rng(0)
    templates: dict[int, torch.Tensor] = {}
    for uid, sessions in enumerate(test_data):
        try:
            templates[uid] = build_user_template(encoder, sessions[:ENROLL_END],
                                                 SEQ_LEN, device, rng=rng)
        except ValueError:
            continue
    LOGGER.info(f"Templates: {len(templates):,} / {len(test_data):,} users "
                f"({len(test_data) - len(templates):,} skipped — short enrollment)")
    streams = load_streams()
    test_streams   = [s for s in streams["test"]        if s["user_idx"] in templates]
    single_streams = [s for s in streams["test_single"] if s["user_idx"] in templates]
    attacks   = [s for s in test_streams if s["stream_type"] == "attack"]
    same_user = [s for s in test_streams if s["stream_type"] == "same_user"]
    LOGGER.info(f"CA eval streams: attack={len(attacks):,}  same_user={len(same_user):,}  single={len(single_streams):,}")
    score_fn = lambda s: score_stream_baseline(s, encoder, templates, SEQ_LEN, device)  # noqa: E731
    ca = Metric.evaluate_streams(score_fn, attacks + same_user, single_streams, n_thresholds=100)
    Metric.print_results(ca, label="Stage 1 baseline (cosine to template)")

    return {"eer": eer, "threshold": thr, "auc": auc, "tar_at_far": tar_far, "ca": ca}


if __name__ == "__main__":
    test(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
