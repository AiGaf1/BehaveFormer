import os
import re
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Iterable

from utils.Config import Config

_DATASET_HPARAMS = (
    "batch_size",
    "epoch_batch_count",
    "keystroke_feature_count",
    "number_of_enrollment_sessions",
    "number_of_verify_sessions",
)

_EPOCH_RE = re.compile(
    r"Epoch No:\s*(?P<epoch>\d+)\s*-\s*Loss:\s*(?P<loss>[-+]?\d*\.?\d+)"
    r"\s*-\s*EER:\s*(?P<eer>[-+]?\d*\.?\d+)\s*-\s*Time:\s*(?P<time>[-+]?\d*\.?\d+)"
)


def load_env_file(env_path: Path | str = ".env") -> None:
    env_file = Path(env_path)
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_run_name(args) -> str:
    return "-".join(part for part in [args.dataset, args.model, args.mode, args.imu] if part)


def build_config(args) -> dict:
    project_config = Config().get_config_dict()
    data_config = project_config["data"]
    hyperparams = project_config["hyperparams"]
    resuming = args.mode == "continue_train" and args.initepoch
    checkpoint_dir = Path(args.script).resolve().parent / "checkpoints"

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "imu": args.imu,
        "metric": args.metric,
        "epochs": _maybe_int(args.epochs),
        "init_epoch": _maybe_int(args.initepoch),
        "testfile": args.testfile,
        "script": str(args.script),
        "gpu": project_config["GPU"],
        "learning_rate": hyperparams["learning_rate"],
        "target_len": hyperparams["target_len"],
        "key_embedding_dim": hyperparams.get("key_embedding_dim"),
        "embedded_keystroke_model_dim": hyperparams.get("embedded_keystroke_model_dim"),
        "keystroke_sequence_len": data_config["keystroke_sequence_len"],
        "resume_from_epoch": _maybe_int(args.initepoch) if resuming else None,
        "resume_from_checkpoint_tar": str(checkpoint_dir / f"training_{args.initepoch}.tar") if resuming else None,
        "resume_from_checkpoint_ckpt": str(checkpoint_dir / f"training_{args.initepoch}.ckpt") if resuming else None,
    }

    if args.dataset:
        config |= {name: hyperparams[name].get(args.dataset) for name in _DATASET_HPARAMS}
        dataset_config = data_config.get(args.dataset)
        if isinstance(dataset_config, dict):
            config["windowing_offset"] = dataset_config.get("windowing_offset")

    if args.imu:
        imu_key = "all" if args.imu == "all" else ("two_types" if "_" in args.imu else "one_type")
        config["imu_sequence_len"] = data_config["imu_sequence_len"]
        config["imu_feature_count"] = hyperparams["imu_feature_count"][imu_key]

    return {k: v for k, v in config.items() if v is not None}


def init_run(
    enabled: bool,
    project: str,
    config: dict,
    run_name: str,
    tags: Iterable[str] | None = None,
    entity: str | None = None,
):
    if not enabled:
        return None
    load_env_file()
    return import_module("wandb").init(
        project=project,
        entity=entity,
        config=config,
        name=run_name,
        tags=list(tags or []),
        save_code=True,
    )


def stream_subprocess(
    command: list[str],
    cwd: Path,
    run=None,
    env: dict | None = None,
    passthrough_output: bool = False,
) -> int:
    if passthrough_output:
        return subprocess.Popen(command, cwd=cwd, env=env).wait()

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in process.stdout or []:
        print(line, end="")
        if run is not None:
            metrics = parse_metrics(line)
            if metrics:
                run.log(metrics, step=metrics["epoch"])

    return process.wait()


def parse_metrics(line: str) -> dict | None:
    match = _EPOCH_RE.search(line)
    if not match:
        return None
    return {
        "epoch": int(match.group("epoch")),
        "train/loss": float(match.group("loss")),
        "val/eer": float(match.group("eer")),
        "epoch_time_seconds": float(match.group("time")),
    }


def _maybe_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
