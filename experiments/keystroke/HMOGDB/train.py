import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BEST_MODELS_DIR = HERE / "best_models"
CHECKPOINTS_DIR = HERE / "checkpoints"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "utils"))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

from model import Model

from experiments.keystroke.nested_training import NestedKeystrokeDatasetSpec, run_nested_keystroke_training_script

HMOG_SPEC = NestedKeystrokeDatasetSpec(
    dataset_key="hmog",
    dataset_dir_name="HMOGDB",
    validation_sequences_per_session=50,
    keystroke_scale_map={index: (255 if index == 9 else 1000) for index in range(10)},
    imu_scale_map={
        0: 10,
        1: 10,
        2: 10,
        3: 1000,
        4: 1000,
        5: 1000,
        15: 1000,
        16: 1000,
        17: 1000,
        24: 100,
        25: 100,
        26: 100,
        27: 10000,
        28: 10000,
        29: 10000,
    },
)


def _make_model(feature_count, seq_len, target_len):
    return Model(feature_count, seq_len, target_len)


if __name__ == "__main__":
    run_nested_keystroke_training_script(
        spec=HMOG_SPEC,
        project_root=PROJECT_ROOT,
        best_model_dir=BEST_MODELS_DIR,
        model_factory=_make_model,
        argv=sys.argv[1:],
    )
