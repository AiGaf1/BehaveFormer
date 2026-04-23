import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

from model import Model  # noqa: E402
from experiments.keystroke_imu_combined.combined_training import CombinedSpec, run_combined_training  # noqa: E402

SPEC = CombinedSpec(
    dataset_key="hmog",
    imu_feature_count_key="two_types",
    imu_columns=slice(None, 24),
)

if __name__ == "__main__":
    run_combined_training(
        spec=SPEC,
        dataset_dir_name="HMOGDB",
        model_factory=Model,
        here=HERE,
        argv=sys.argv[1:],
    )
