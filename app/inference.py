"""Backward-compatibility shim — kept so existing call sites that do
`from app.inference import AuthEngine, calibrate_threshold, …` keep working.

New code should import directly from `app.engine` or `app.calibration`.
"""

from app.calibration import (
    build_bank_windows,
    calibrate_threshold,
    save_bank,
    score_distributions,
)
from app.engine import (
    BANK_DIR,
    BANK_PATH,
    AuthEngine,
    keystrokes_to_features,
)

__all__ = [
    "AuthEngine",
    "BANK_DIR",
    "BANK_PATH",
    "build_bank_windows",
    "calibrate_threshold",
    "keystrokes_to_features",
    "save_bank",
    "score_distributions",
]
