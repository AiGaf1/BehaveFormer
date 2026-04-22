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

HUMI_SPEC = NestedKeystrokeDatasetSpec(
    dataset_key="humi",
    dataset_dir_name="HuMIdb",
    validation_sequences_per_session=1,
    keystroke_columns=(1, 2, 3),
    convert_to_float=True,
    keystroke_scale_map={
        1: 1000,
        2: 1000,
        3: 255,
    },
    imu_scale_map={
        0: 10,
        1: 10,
        2: 10,
        3: 1000,
        4: 1000,
        5: 1000,
        15: 100,
        16: 100,
        17: 100,
        24: 1000,
        25: 1000,
        26: 1000,
        27: 1000,
        28: 1000,
        29: 1000,
    },
)


def _make_model(feature_count, seq_len, target_len):
    return Model(feature_count, seq_len, target_len)


if __name__ == "__main__":
    run_nested_keystroke_training_script(
        spec=HUMI_SPEC,
        project_root=PROJECT_ROOT,
        best_model_dir=BEST_MODELS_DIR,
        model_factory=_make_model,
        argv=sys.argv[1:],
    )