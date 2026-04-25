import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "utils"))
from Config import Config

from experiments.common.logger import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_FILE = PROJECT_ROOT / "download"
DATASET_DIR   = PROJECT_ROOT / "dataset"

IMU_PREFIXES  = ["a", "g", "m"]
IMU_COLUMNS   = ["x", "y", "z", "fft_x", "fft_y", "fft_z", "fd_x", "fd_y", "fd_z", "sd_x", "sd_y", "sd_z"]
IMU_FULL_COLS = [f"{p}_{c}" for p in IMU_PREFIXES for c in IMU_COLUMNS]


_PREP_DIR = Path(__file__).resolve().parent / "prep_data"


def maybe_download(file_id: str, target_path: Path) -> None:
    if target_path.exists() or not file_id:
        return
    subprocess.run(f"gdown {file_id}", shell=True, check=True, cwd=target_path.parent)


# ── Download / extraction ────────────────────────────────────────────────────

def download_dataset_file(dataset_url, output_path=DOWNLOAD_FILE):
    if not dataset_url:
        raise ValueError("HMOG dataset_url is empty. Add a valid download URL in config.json.")
    if output_path.exists() and output_path.stat().st_size > 0:
        LOGGER.info("Using existing dataset archive: %s", output_path)
        return
    subprocess.run(["wget", "-c", "-O", str(output_path), dataset_url], check=True)
    LOGGER.info("Download completed!")


def extract_zip(zipped_file, to_folder):
    if shutil.which("7z") is None:
        raise RuntimeError("7z is required. Install it with: sudo apt-get install -y p7zip-full")
    subprocess.run(["7z", "x", "-y", f"-o{to_folder}", str(zipped_file)], check=True)
    for file_path in Path(to_folder).rglob("*.zip"):
        subprocess.run(["7z", "x", "-y", f"-o{file_path.parent}", str(file_path)], check=True)


def extract(absolute_path):
    LOGGER.info("Extracting...")
    extract_zip(Path(absolute_path), DATASET_DIR)
    LOGGER.info("File is unzipped in %s", DATASET_DIR)
    return DATASET_DIR


def get_filtered_users():
    shutil.rmtree(DATASET_DIR / "public_dataset" / "733162", ignore_errors=True)


# ── Keystroke processing ─────────────────────────────────────────────────────

def check_couple_order(dataset, index):
    """Ensure a key-down event precedes its key-up partner at `index`."""
    press_first = dataset.at[index, "press_type"] == 0

    if press_first:
        if dataset.at[index, "event_time"] < dataset.at[index + 1, "event_time"]:
            dataset.at[index, "press_type"] = 1
            dataset.at[index + 1, "press_type"] = 0
        else:
            dataset.loc[index], dataset.loc[index + 1] = dataset.loc[index + 1].copy(), dataset.loc[index].copy()
    else:
        if dataset.at[index, "event_time"] > dataset.at[index + 1, "event_time"]:
            dataset.at[index, "press_type"] = 0
            dataset.at[index + 1, "press_type"] = 1
            dataset.loc[index], dataset.loc[index + 1] = dataset.loc[index + 1].copy(), dataset.loc[index].copy()


def generate_couple(datasets):
    """Pair every key-down event with a matching key-up event."""
    for df in datasets:
        index = 0
        while index < len(df):
            last = index == len(df) - 1
            mismatched = not last and (
                df.at[index, "key_code"] != df.at[index + 1, "key_code"]
                or df.at[index, "press_type"] == df.at[index + 1, "press_type"]
            )

            if last or mismatched:
                df.loc[index + 0.5] = [
                    df.at[index, "event_time"],
                    1 - int(df.at[index, "press_type"]),
                    df.at[index, "key_code"],
                    df.at[index, "user_id"],
                ]
                df.sort_index(inplace=True)
                df.reset_index(drop=True, inplace=True)
                check_couple_order(df, index)
                if last:
                    break

            check_couple_order(df, index)
            index += 2


def shrink_couple_data(keystroke_df):
    """Collapse paired (press, release) rows into a single row with both timestamps."""
    if len(keystroke_df) % 2 != 0:
        raise ValueError("Keystroke data must contain an even number of rows (press/release pairs).")

    release_times = keystroke_df.iloc[1::2]["event_time"].values
    keystroke_df = keystroke_df.iloc[::2].copy()
    keystroke_df["release_time"] = release_times
    keystroke_df.rename(columns={"event_time": "press_time"}, inplace=True)
    keystroke_df.drop(columns=["press_type"], inplace=True)
    keystroke_df.reset_index(drop=True, inplace=True)
    return keystroke_df


# ── Feature extraction ───────────────────────────────────────────────────────

def keystroke_feature_extract(keystroke, key_vocab: dict):
    hold_time   = (keystroke["release_time"] - keystroke["press_time"]).values / 1000
    flight_time = np.zeros(len(keystroke), dtype=np.float32)
    flight_time[:-1] = (keystroke["press_time"].values[1:] - keystroke["release_time"].values[:-1]) / 1000
    key_code = np.array([key_vocab[k] for k in keystroke["key_code"].values], dtype=np.float32)
    keystroke["hold_time"]   = hold_time
    keystroke["flight_time"] = flight_time
    keystroke["key"]         = key_code
    return keystroke


def imu_feature_extract(imu_df):
    for axis in ["x", "y", "z"]:
        imu_df[f"fft_{axis}"] = np.abs(np.fft.fft(imu_df[axis].values))
        imu_df[f"fd_{axis}"]  = np.gradient(imu_df[axis].values, edge_order=2)
        imu_df[f"sd_{axis}"]  = np.gradient(imu_df[f"fd_{axis}"].values, edge_order=2)
    return imu_df


# ── IMU syncing ──────────────────────────────────────────────────────────────

def embed_zero_padding(sequence, target_length):
    shortfall = target_length - len(sequence)
    if shortfall <= 0:
        return sequence
    padding = pd.DataFrame(
        np.zeros((shortfall, sequence.shape[1])),
        columns=sequence.columns
    )
    return pd.concat([sequence, padding], ignore_index=True)


def sync_imu_data(acc, gyr, mag, sync_period, imu_sequence_length):
    for name, df in [("accelerometer", acc), ("gyroscope", gyr), ("magnetometer", mag)]:
        if df.isnull().values.any():
            LOGGER.warning("%s dataframe contains NaN", name)
            df.fillna(0, inplace=True)

    start_time   = min(df["event_time"].iloc[0]  for df in (acc, gyr, mag) if not df.empty)
    highest_time = max(df["event_time"].iloc[-1] for df in (acc, gyr, mag) if not df.empty)

    bins = np.arange(start_time, highest_time + sync_period, sync_period)
    n_windows = len(bins) - 1

    parts = []
    prefixes = {"a": acc, "g": gyr, "m": mag}
    for prefix, df in prefixes.items():
        labels = pd.cut(df["event_time"], bins=bins, labels=False, right=True)
        empty_windows = n_windows - labels.dropna().nunique()
        if empty_windows > 0:
            LOGGER.warning("%d windows have no IMU elements for sensor %s", empty_windows, prefix)
        cols = [c for c in IMU_COLUMNS if c in df.columns]
        part = (
            df[cols].groupby(labels, observed=False).mean()
            .reindex(range(n_windows))
            .fillna(0.0)
        )
        part.columns = [f"{prefix}_{c}" for c in cols]
        parts.append(part)

    imu_sequence = pd.concat(parts, axis=1)[IMU_FULL_COLS]

    if len(imu_sequence) > imu_sequence_length:
        imu_sequence = imu_sequence.iloc[:imu_sequence_length]
    else:
        imu_sequence = embed_zero_padding(imu_sequence, imu_sequence_length)

    return imu_sequence


# ── Windowed pre-processing ──────────────────────────────────────────────────

def pre_process(event_data, event_seq_len, imu_seq_len, offset, acc, gyr, mag):
    # Slice keystroke windows
    n = len(event_data)
    event_sequences = []
    start = 0
    while start < n:
        end = start + event_seq_len
        chunk = event_data.loc[start:, :] if end >= n else event_data.loc[start:end - 1, :]
        event_sequences.append(chunk.reset_index(drop=True))
        if end >= n:
            break
        start += offset

    # Align IMU windows to each keystroke window
    imu_sequences = []
    for seq in event_sequences:
        start_t = int(seq.iloc[0]["press_time"])
        end_t   = int(seq.iloc[-1]["release_time"])
        period  = (end_t - start_t) / imu_seq_len

        imu_sequences.append(sync_imu_data(
            acc.loc[(acc["event_time"] >= start_t) & (acc["event_time"] <= end_t)],
            gyr.loc[(gyr["event_time"] >= start_t) & (gyr["event_time"] <= end_t)],
            mag.loc[(mag["event_time"] >= start_t) & (mag["event_time"] <= end_t)],
            period, imu_seq_len,
        ))

    event_sequences[-1] = embed_zero_padding(event_sequences[-1], event_seq_len)
    return event_sequences, imu_sequences


# ── Dataset reading ──────────────────────────────────────────────────────────

def get_users(absolute_path):
    public_dir = Path(absolute_path) / "public_dataset"
    return [d for d in os.listdir(public_dir) if d.isnumeric() and "." not in d]


def build_key_vocab(absolute_path, users_list) -> dict:
    """Scan all sessions and return a mapping raw_key_code -> 1-based index (0 reserved for padding)."""
    public_dir = Path(absolute_path) / "public_dataset"
    raw_keys = set()
    for userid in users_list:
        for session in os.listdir(public_dir / userid):
            if "." in session:
                continue
            ks = pd.read_csv(public_dir / userid / session / "KeyPressEvent.csv",
                             header=None, usecols=[4], names=["key_code"])
            raw_keys.update(ks["key_code"].unique())
    return {k: i + 1 for i, k in enumerate(sorted(raw_keys))}


def read_keystroke(absolute_path, users_list, keystroke_sequence_len, imu_sequence_len, windowing_offset, key_vocab: dict):
    public_dir = Path(absolute_path) / "public_dataset"
    all_data = []

    for i, userid in enumerate(users_list, 1):
        session_data = []
        for session in os.listdir(public_dir / userid):
            if "." in session:
                continue

            session_dir = public_dir / userid / session
            ks = pd.read_csv(session_dir / "KeyPressEvent.csv", header=None,
                             usecols=[0, 3, 4], names=["event_time", "press_type", "key_code"])
            if ks.empty:
                continue

            ks["user_id"] = f"user_{userid}"
            generate_couple([ks])
            ks.drop(columns=["user_id"], inplace=True)
            ks = shrink_couple_data(ks)       # now returns instead of mutating
            ks = keystroke_feature_extract(ks, key_vocab)

            def read_imu(filename):
                return imu_feature_extract(
                    pd.read_csv(session_dir / filename, header=None,
                                usecols=[0, 3, 4, 5], names=["event_time", "x", "y", "z"])
                )

            acc = read_imu("Accelerometer.csv")
            gyr = read_imu("Gyroscope.csv")
            mag = read_imu("Magnetometer.csv")

            ks_seqs, imu_seqs = pre_process(ks, keystroke_sequence_len, imu_sequence_len,
                                            windowing_offset, acc, gyr, mag)

            sequences = [
                [seq[["hold_time", "flight_time", "key"]].to_numpy(dtype=np.float32),
                 imu.to_numpy()]
                for seq, imu in zip(ks_seqs, imu_seqs)
            ]
            session_data.append(sequences)
            LOGGER.info("Session %s completed", session)

        all_data.append(session_data)
        LOGGER.info("User %s completed (%s/%s)", userid, i, len(users_list))

    return all_data


# ── Pickle helpers ───────────────────────────────────────────────────────────

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def build_pickles(keystroke_sequence_len: int, imu_sequence_len: int, windowing_offset: int) -> None:
    """Build and save train/val/test pickles from the raw public_dataset directory."""
    raw_dir = _PREP_DIR / "public_dataset"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_dir}. Run build.py to download and extract it first."
        )

    users_list = get_users(raw_dir.parent)
    key_vocab = build_key_vocab(raw_dir.parent, users_list)
    LOGGER.info("Key vocabulary size: %s", len(key_vocab))
    save_pickle(key_vocab, _PREP_DIR / "key_vocab.pickle")

    train_users, val_test = train_test_split(users_list, test_size=30, train_size=69, shuffle=True)
    val_users, test_users = train_test_split(val_test,   test_size=15, train_size=15, shuffle=True)

    for split, users in [("training", train_users), ("validation", val_users), ("testing", test_users)]:
        save_pickle(
            read_keystroke(raw_dir.parent, users, keystroke_sequence_len, imu_sequence_len, windowing_offset, key_vocab),
            _PREP_DIR / f"{split}_keystroke_imu_data_all.pickle",
        )


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_data = Config().get_config_dict()["data"]

    keystroke_sequence_len = config_data["keystroke_sequence_len"] or 50
    imu_sequence_len       = config_data["imu_sequence_len"]       or 100
    windowing_offset       = config_data["hmog"]["windowing_offset"] or 5

    pickles = [_PREP_DIR / f"{s}_keystroke_imu_data_all.pickle" for s in ("training", "validation", "testing")]
    if all(p.exists() for p in pickles):
        LOGGER.info("All HMOG pickles already exist, skipping build.")
    else:
        dataset_url = config_data["hmog"]["dataset_url"]
        download_dataset_file(dataset_url)
        extract(DOWNLOAD_FILE)
        get_filtered_users()
        build_pickles(
            keystroke_sequence_len=keystroke_sequence_len,
            imu_sequence_len=imu_sequence_len,
            windowing_offset=windowing_offset,
        )