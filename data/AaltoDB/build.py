import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "utils"))
from Config import Config

_HERE = Path(__file__).resolve().parent
_CSV  = _HERE / "prep_data" / "keystroke_data.csv"


def maybe_download(gdown_id: str) -> None:
    if gdown_id and not _CSV.exists():
        subprocess.run(f"gdown {gdown_id}", shell=True)


if __name__ == "__main__":
    dataset_url = Config().get_config_dict()["data"]["aalto"]["dataset_url"]

    if shutil.which("7z") is None:
        raise RuntimeError("7z is required. Install it with: sudo apt-get install -y p7zip-full")

    subprocess.run(["wget", "-c", dataset_url], check=True)
    subprocess.run(["7z", "x", "-y", "csv_raw_and_processed.zip"], check=True)

    raw_dir = _HERE / "Data_Raw"
    headers = pd.read_csv(raw_dir / "test_sections_header.csv").Field.tolist()
    test_sections = pd.read_csv(raw_dir / "test_sections.csv", names=headers, encoding="latin-1", on_bad_lines="skip")
    test_sections = test_sections[["TEST_SECTION_ID", "PARTICIPANT_ID"]]

    headers = pd.read_csv(raw_dir / "keystrokes_header.csv").Field.tolist()
    chunks = pd.read_csv(raw_dir / "keystrokes.csv", names=headers, encoding="latin-1", chunksize=10_000_000, on_bad_lines="skip")
    keystroke = pd.concat(
        chunk[["TEST_SECTION_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE"]].merge(test_sections)
        for chunk in chunks
    )
    keystroke.rename(columns={
        "TEST_SECTION_ID": "session_id",
        "PARTICIPANT_ID":  "user_id",
        "PRESS_TIME":      "press_time",
        "RELEASE_TIME":    "release_time",
        "KEYCODE":         "key_code",
    }, inplace=True)
    keystroke.to_csv(_CSV)
