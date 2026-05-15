"""Shared checkpoint and DataLoader utilities for AaltoDB experiments."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.AaltoDB.prepare import load as load_aalto
from experiments.config import parse_runtime_config
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size
from utils.config import seed_worker


def latest_ckpt(directory: Path) -> Path:
    ckpts = sorted(directory.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {directory}")
    return ckpts[-1]


def load_encoder(ckpt_path: Path | None, device: torch.device):
    from experiments.stage1_encoder.model import Encoder
    from experiments.stage1_encoder.test import _pick_best_ckpt
    from experiments.stage1_encoder.train import (
        DROPOUT, HEADS, KEY_EMB, LFF_FEATURES, NUM_LAYERS, SEQ_LEN,
    )
    if ckpt_path is None:
        ckpt_path = _pick_best_ckpt()
    train_data, _, test_data = load_aalto()
    enc = Encoder(
        seq_len=SEQ_LEN,
        vocab_size=aalto_vocab_size(train_data),
        key_emb=KEY_EMB,
        feature_ranges=aalto_feature_ranges(train_data),
        lff_features=LFF_FEATURES,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")})
    enc.eval()
    return enc, test_data


def load_detector(ckpt_path: Path, device: torch.device, embed_dim: int):
    from experiments.stage2_detector.model import BankAwareDetector
    from experiments.stage1_encoder.train import DROPOUT
    from experiments.stage2_detector.train import HIDDEN_DIM, TOP_K_STATS
    det = BankAwareDetector(
        embed_dim=embed_dim, top_k=TOP_K_STATS,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    det.load_state_dict({k.removeprefix("detector."): v for k, v in state.items() if k.startswith("detector.")})
    det.eval()
    return det


def make_dataloader(dataset, batch_size: int, shuffle: bool, config, seed_offset: int = 0,
                    collate_fn=None):
    project_root = Path(__file__).resolve().parents[1]
    project_cfg  = json.loads((project_root / "config.json").read_text())
    cfg = parse_runtime_config(project_cfg)
    g = torch.Generator()
    g.manual_seed(cfg.seed + seed_offset)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, **kwargs)
