import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from utils.wandb import build_config, build_run_name, init_run, load_env_file, stream_subprocess

ROOT = Path(__file__).parent.resolve()

EXPERIMENTS = {
    "keystroke": ROOT / "experiments/keystroke",
    "keystroke_imu": ROOT / "experiments/keystroke_imu_combined",
    "tl": ROOT / "experiments/transfer_learning",
}

DATASETS = {
    "aalto": "AaltoDB",
    "hmog": "HMOGDB",
    "humi": "HuMIdb",
}

IMU_MODES = ["acc", "gyr", "mag", "acc_gyr", "acc_mag", "mag_gyr", "all"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*EXPERIMENTS, None])
    parser.add_argument("--mode", choices=["preprocess", "train", "continue_train", "test"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--metric", choices=["basic", "det", "pca"])
    parser.add_argument("--testfile")
    parser.add_argument("--epochs")
    parser.add_argument("--initepoch")
    parser.add_argument("--imu", choices=IMU_MODES)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="BehaveFormer")
    parser.add_argument("--wandb-entity")
    return parser.parse_args()


def extra_args(args):
    if args.mode == "test":
        return [args.metric, args.testfile]
    if args.mode == "train":
        return [args.epochs]
    if args.mode == "continue_train":
        return [args.epochs, args.initepoch]
    return []


def resolve_script(args):
    dataset = DATASETS[args.dataset]

    if args.model is None:
        return ROOT / "data" / dataset / f"{args.mode}.py"

    base = EXPERIMENTS[args.model]

    if args.model == "tl":
        if args.mode in {"train", "continue_train"}:
            return base / f"train_{dataset}.py"
        return base / f"{args.mode}.py"

    experiment_dir = base / dataset
    if args.imu:
        experiment_dir = experiment_dir / f"imu_{args.imu}"

    script = "train" if args.mode == "continue_train" else args.mode
    return experiment_dir / f"{script}.py"


def use_native_wandb(args) -> bool:
    return (
        args.wandb
        and args.model == "keystroke"
        and args.dataset == "aalto"
        and args.mode in {"train", "continue_train"}
    )


def subprocess_env_for_native_wandb(args, config: dict, tags: list[str]) -> dict:
    load_env_file()

    env = os.environ.copy()
    env.update(
        {
            "BEHAVEFORMER_WANDB_ENABLED": "1",
            "BEHAVEFORMER_WANDB_PROJECT": args.wandb_project,
            "BEHAVEFORMER_WANDB_ENTITY": args.wandb_entity or "",
            "BEHAVEFORMER_WANDB_RUN_NAME": build_run_name(args),
            "BEHAVEFORMER_WANDB_CONFIG_JSON": json.dumps(config),
            "BEHAVEFORMER_WANDB_TAGS_JSON": json.dumps(tags),
        }
    )
    return env


if __name__ == "__main__":
    args = parse_args()
    script = resolve_script(args)
    args.script = script

    command = [sys.executable, str(script), *[str(arg) for arg in extra_args(args) if arg is not None]]
    config = build_config(args)
    tags = [args.dataset, args.mode]
    if args.model:
        tags.append(args.model)
    if args.imu:
        tags.append(args.imu)

    run = None
    subprocess_env = None
    if use_native_wandb(args):
        subprocess_env = subprocess_env_for_native_wandb(args, config, tags)
    else:
        run = init_run(
            enabled=args.wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            run_name=build_run_name(args),
            tags=tags,
        )

    exit_code = stream_subprocess(command, cwd=ROOT, run=run, env=subprocess_env)
    if run is not None:
        run.finish()

    sys.exit(exit_code)
