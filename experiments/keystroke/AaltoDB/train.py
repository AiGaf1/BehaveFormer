import json, logging, math, os, pickle, subprocess, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "AaltoDB" / "prep_data"
BEST_MODELS_DIR = HERE / "best_models"
CHECKPOINTS_DIR = HERE / "checkpoints"

CSV_PATH             = DATA_DIR / "keystroke_data.csv"
TRAINING_PICKLE      = DATA_DIR / "training_data.pickle"
VALIDATION_PICKLE    = DATA_DIR / "validation_data.pickle"
TESTING_PICKLE       = DATA_DIR / "testing_data.pickle"
TRAINING_FEATURES_PICKLE   = DATA_DIR / "training_features.pickle"
VALIDATION_FEATURES_PICKLE = DATA_DIR / "validation_features.pickle"

SESSIONS_PER_USER = 15
ENROLL_SESSIONS   = 10
VERIFY_SESSIONS   = 5

for p in ["utils", "evaluation", ""]:
    sys.path.append(str(PROJECT_ROOT / p))

from Config import Config
from metrics import Metric
from model import KeystrokeTransformer
from experiments.keystroke.common.loss import TripletLoss

warnings.filterwarnings("ignore", message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*")
logging.getLogger("pytorch_lightning.trainer.connectors.logger_connector.logger_connector").addFilter(
    type("F", (logging.Filter,), {"filter": lambda self, r: "try installing [litlogger]" not in r.getMessage()})()
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_pickle(path, obj):
    with open(path, "wb") as f: pickle.dump(obj, f)

def _load_pickle(path):
    with open(path, "rb") as f: return pickle.load(f)

def _tensor_float(v):
    return None if v is None else float(v.detach().cpu() if isinstance(v, torch.Tensor) else v)

def _move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): state[k] = v.to(device)

def _pad_sequence(seq, seq_len):
    if len(seq) == seq_len: return seq
    if len(seq) < seq_len:  return np.vstack([seq, np.zeros((seq_len - len(seq), seq.shape[1]))])
    return seq[:seq_len]

def _recommended_num_workers():
    return max(1, min(8, (os.cpu_count() or 1) - 1))

def _json_env(name, default):
    raw = os.getenv(name)
    try: return json.loads(raw) if raw else default
    except json.JSONDecodeError: return default

# ── Data preprocessing ────────────────────────────────────────────────────────

def _compute_features(seq):
    n = len(seq)
    features = np.zeros((n, 10))
    for i, (press, release, key) in enumerate(seq):
        nxt  = seq[i + 1] if i < n - 1 else None
        nxt2 = seq[i + 2] if i < n - 2 else None
        features[i] = [
            (release - press) / 1000,
            (nxt[0]  - release) / 1000 if nxt  else 0.0,
            (nxt[0]  - press)   / 1000 if nxt  else 0.0,
            (nxt[1]  - release) / 1000 if nxt  else 0.0,
            (nxt[1]  - press)   / 1000 if nxt  else 0.0,
            (nxt2[0] - release) / 1000 if nxt2 else 0.0,
            (nxt2[0] - press)   / 1000 if nxt2 else 0.0,
            (nxt2[1] - release) / 1000 if nxt2 else 0.0,
            (nxt2[1] - press)   / 1000 if nxt2 else 0.0,
            key / 255,
        ]
    return features

def preprocess():
    data = pd.read_csv(CSV_PATH)
    assert not data.isnull().values.any()
    data_dict = {
        user: [g[["press_time", "release_time", "key_code"]].to_numpy() for _, g in s.groupby("session_id")]
        for user, s in data.groupby("user_id")
    }
    all_users = [s for s in data_dict.values() if len(s) == SESSIONS_PER_USER]
    print(f"Users after filtering: {len(all_users)}")
    for pickle_path, split in zip(
        [TRAINING_PICKLE, VALIDATION_PICKLE, TESTING_PICKLE],
        [all_users[:-1050], all_users[-1050:-1000], all_users[-1000:]],
    ):
        _save_pickle(pickle_path, split)

def _ensure_preprocessed_pickles():
    if all(p.exists() for p in [TRAINING_PICKLE, VALIDATION_PICKLE, TESTING_PICKLE]):
        print("Using cached Aalto split pickles")
    else:
        print("Building Aalto split pickles")
        preprocess()

def _load_feature_pickles():
    if TRAINING_FEATURES_PICKLE.exists() and VALIDATION_FEATURES_PICKLE.exists():
        print("Using cached Aalto feature pickles")
        return _load_pickle(TRAINING_FEATURES_PICKLE), _load_pickle(VALIDATION_FEATURES_PICKLE)
    print("Building Aalto feature pickles")
    training_data, validation_data = _load_pickle(TRAINING_PICKLE), _load_pickle(VALIDATION_PICKLE)
    for dataset in (training_data, validation_data):
        for user_seqs in dataset:
            for i, seq in enumerate(user_seqs):
                user_seqs[i] = _compute_features(seq)
    _save_pickle(TRAINING_FEATURES_PICKLE, training_data)
    _save_pickle(VALIDATION_FEATURES_PICKLE, validation_data)
    return training_data, validation_data

# ── Datasets ──────────────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    def __init__(self, data, batch_size, epoch_batch_count, seq_len):
        self.data, self.batch_size, self.epoch_batch_count, self.seq_len = data, batch_size, epoch_batch_count, seq_len

    def __len__(self): return self.batch_size * self.epoch_batch_count

    def __getitem__(self, _):
        genuine, imposter = np.random.choice(len(self.data), size=2, replace=False)
        s1, s2 = np.random.choice(SESSIONS_PER_USER, size=2, replace=False)
        s3 = np.random.randint(SESSIONS_PER_USER)
        return tuple(_pad_sequence(self.data[u][s], self.seq_len)
                     for u, s in [(genuine, s1), (genuine, s2), (imposter, s3)])


class TestDataset(Dataset):
    def __init__(self, data, seq_len): self.data, self.seq_len = data, seq_len
    def __len__(self): return len(self.data) * SESSIONS_PER_USER
    def __getitem__(self, idx):
        return _pad_sequence(self.data[idx // SESSIONS_PER_USER][idx % SESSIONS_PER_USER], self.seq_len)

# ── Data module ───────────────────────────────────────────────────────────────

class AaltoDataModule(pl.LightningDataModule):
    def __init__(self, training_data, validation_data, batch_size, epoch_batch_count, seq_len, num_workers, pin_memory):
        super().__init__()
        self.__dict__.update(locals())  # store all args as attributes

    def _loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(TrainDataset(self.training_data, self.batch_size, self.epoch_batch_count, self.seq_len))

    def val_dataloader(self):
        return self._loader(TestDataset(self.validation_data, self.seq_len))

# ── Lightning module ──────────────────────────────────────────────────────────

def _make_transformer(feature_count, seq_len, target_len):
    return KeystrokeTransformer(6, feature_count, 20, 5, 10, seq_len, target_len, 0.1)

class KeystrokeLightningModule(pl.LightningModule):
    def __init__(self, feature_count, seq_len, target_len, learning_rate, validation_user_count):
        super().__init__()
        self.save_hyperparameters()
        self.model = _make_transformer(feature_count, seq_len, target_len)
        self.loss_fn = TripletLoss()
        self.resume_optimizer_state = None
        self.validation_embeddings = []

    def training_step(self, batch, _):
        anchor, positive, negative = batch
        loss = self.loss_fn(*[self.model(x.float()) for x in (anchor, positive, negative)])
        self.log("train_loss_step", loss, on_step=True,  on_epoch=False, prog_bar=True,  logger=False, batch_size=anchor.size(0))
        self.log("train_loss",      loss, on_step=False, on_epoch=True,  prog_bar=False, logger=True,  batch_size=anchor.size(0))
        return loss

    def on_validation_epoch_start(self): self.validation_embeddings = []

    def validation_step(self, batch, _):
        self.validation_embeddings.append(self.model(batch.float()).detach())

    def on_validation_epoch_end(self):
        if not self.validation_embeddings: return
        embeddings = torch.cat(self.validation_embeddings).view(
            self.hparams.validation_user_count, SESSIONS_PER_USER, self.hparams.target_len)
        eer = Metric.cal_user_eer_aalto(embeddings, ENROLL_SESSIONS, VERIFY_SESSIONS)[0]
        self.log("val_eer", eer, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def on_fit_start(self):
        if self.resume_optimizer_state is None: return
        optimizer = self.optimizers()
        optimizer.load_state_dict(self.resume_optimizer_state)
        _move_optimizer_to_device(optimizer, self.device)
        self.resume_optimizer_state = None

    def export_model(self):
        exported = _make_transformer(self.hparams.feature_count, self.hparams.seq_len, self.hparams.target_len)
        exported.load_state_dict({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
        exported.train(self.model.training)
        return exported

# ── Callback ──────────────────────────────────────────────────────────────────

class TrainingArtifactsCallback(Callback):
    def __init__(self, best_model_dir, checkpoint_dir, epoch_offset=0, best_eer=math.inf,
                 wandb_logger=None, resume_from_epoch=None):
        super().__init__()
        self.best_model_dir, self.checkpoint_dir = Path(best_model_dir), Path(checkpoint_dir)
        self.epoch_offset, self.best_eer = epoch_offset, best_eer
        self.wandb_logger, self.resume_from_epoch = wandb_logger, resume_from_epoch
        self._epoch_start = self._last_epoch = epoch_offset
        self._saved_checkpoint_epochs = set()

    def state_dict(self): return {"best_eer": self.best_eer}
    def load_state_dict(self, state): self.best_eer = state.get("best_eer", math.inf)
    def on_train_epoch_start(self, trainer, pl_module): self._epoch_start = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = _tensor_float(trainer.callback_metrics.get("train_loss"))
        eer  = _tensor_float(trainer.callback_metrics.get("val_eer"))
        if loss is None or eer is None: return

        epoch = self.epoch_offset + trainer.current_epoch + 1
        self._last_epoch = epoch
        elapsed = 0.0 if self._epoch_start is None else time.time() - self._epoch_start
        print(f"------> Epoch No: {epoch} - Loss: {loss:>7f} - EER: {eer:>4f} - Time: {elapsed:>2f}")

        if eer < self.best_eer:
            model_path = self.best_model_dir / f"epoch_{epoch}_eer_{eer}.pt"
            torch.save(pl_module.export_model(), model_path)
            print(f"Model saved - EER improved from {self.best_eer} to {eer}")
            self.best_eer = eer
            self._log_artifact("model", model_path, epoch, eer, aliases=["best", f"epoch-{epoch}"])

        if epoch % 50 == 0: self._save_checkpoint(trainer, pl_module, epoch)

    def on_train_end(self, trainer, pl_module):
        if self._last_epoch > 0: self._save_checkpoint(trainer, pl_module, self._last_epoch)

    def _save_checkpoint(self, trainer, pl_module, epoch):
        if epoch in self._saved_checkpoint_epochs: return
        ckpt_path = self.checkpoint_dir / f"training_{epoch}"
        ckpt_file, tar_file = ckpt_path.with_suffix(".ckpt"), ckpt_path.with_suffix(".tar")
        trainer.save_checkpoint(str(ckpt_file))
        torch.save({"epoch": epoch, "model_state_dict": pl_module.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizers[0].state_dict(), "eer": self.best_eer}, tar_file)
        self._saved_checkpoint_epochs.add(epoch)
        self._log_artifact("checkpoint", ckpt_file, epoch, self.best_eer,
                            aliases=["latest", f"epoch-{epoch}"], extra_file=tar_file)

    def _log_artifact(self, artifact_type, path, epoch, eer, aliases, extra_file=None):
        if self.wandb_logger is None: return
        meta = {k: v for k, v in {"epoch": epoch, "val_eer": eer, "dataset": "aalto", "model": "keystroke",
                                   "resume_from_epoch": self.resume_from_epoch}.items() if v is not None}
        artifact = wandb.Artifact(
            name=f"aalto-keystroke-{artifact_type}-{self.wandb_logger.experiment.id}",
            type=artifact_type, metadata=meta)
        artifact.add_file(str(path))
        if extra_file: artifact.add_file(str(extra_file))
        self.wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

# ── WandB setup ───────────────────────────────────────────────────────────────

def _build_wandb_logger():
    if os.getenv("BEHAVEFORMER_WANDB_ENABLED") != "1": return None
    logger = WandbLogger(
        project=os.getenv("BEHAVEFORMER_WANDB_PROJECT") or "BehaveFormer",
        entity=os.getenv("BEHAVEFORMER_WANDB_ENTITY") or None,
        name=os.getenv("BEHAVEFORMER_WANDB_RUN_NAME") or None,
        tags=_json_env("BEHAVEFORMER_WANDB_TAGS_JSON", []),
        log_model=False, save_dir=str(PROJECT_ROOT),
    )
    logger.experiment.config.update(_json_env("BEHAVEFORMER_WANDB_CONFIG_JSON", {}), allow_val_change=True)
    return logger

def _log_model_summary(wandb_logger, model):
    if wandb_logger is None: return
    wandb_logger.experiment.config.update({
        "model_class_name": model.__class__.__name__,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }, allow_val_change=True)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = Config().get_config_dict()

    gdown_id = config["preprocessed_data"]["aalto"]["keystroke"]
    if gdown_id and not CSV_PATH.exists():
        subprocess.run(f"gdown {gdown_id}", shell=True)

    _ensure_preprocessed_pickles()
    training_data, validation_data = _load_feature_pickles()

    hp, data = config["hyperparams"], config["data"]
    batch_size        = hp["batch_size"]["aalto"]
    epoch_batch_count = hp["epoch_batch_count"]["aalto"]
    seq_len           = data["keystroke_sequence_len"]
    feature_count     = hp["keystroke_feature_count"]["aalto"]
    target_len        = hp["target_len"]
    learning_rate     = hp["learning_rate"]

    BEST_MODELS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    epochs       = int(sys.argv[1])
    epoch_offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    best_eer, model_state, optim_state = math.inf, None, None

    if epoch_offset:
        ckpt = torch.load(CHECKPOINTS_DIR / f"training_{epoch_offset}.tar", map_location="cpu")
        model_state, optim_state, best_eer = ckpt["model_state_dict"], ckpt["optimizer_state_dict"], ckpt["eer"]

    use_gpu = config["GPU"] == "True" and torch.cuda.is_available()
    if use_gpu: torch.set_float32_matmul_precision("high")

    module = KeystrokeLightningModule(feature_count, seq_len, target_len, learning_rate, len(validation_data))
    if model_state:
        module.model.load_state_dict(model_state)
        module.resume_optimizer_state = optim_state

    wandb_logger = _build_wandb_logger()
    _log_model_summary(wandb_logger, module.model)
    if wandb_logger: wandb_logger.watch(module.model, log="all", log_freq=10)

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu", devices=1, max_epochs=epochs,
        callbacks=[TrainingArtifactsCallback(BEST_MODELS_DIR, CHECKPOINTS_DIR, epoch_offset, best_eer,
                                             wandb_logger=wandb_logger,
                                             resume_from_epoch=epoch_offset or None)],
        logger=wandb_logger or False, log_every_n_steps=10,
        enable_checkpointing=False, enable_progress_bar=True,
        enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=AaltoDataModule(
        training_data, validation_data, batch_size, epoch_batch_count,
        seq_len, num_workers=_recommended_num_workers(), pin_memory=use_gpu,
    ))