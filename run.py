import argparse
import subprocess
import sys
from pathlib import Path


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


if __name__ == "__main__":
    args = parse_args()
    script = resolve_script(args)
    command = [sys.executable, str(script), *[str(arg) for arg in extra_args(args) if arg is not None]]
    subprocess.run(command)
