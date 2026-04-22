import math
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2] / "utils"))
from Config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_FILE = PROJECT_ROOT / "download"
DATASET_DIR   = PROJECT_ROOT / "dataset"

IMU_PREFIXES  = ["a", "g", "m"]
IMU_COLUMNS   = ["x", "y", "z", "fft_x", "fft_y", "fft_z", "fd_x", "fd_y", "fd_z", "sd_x", "sd_y", "sd_z"]
IMU_FULL_COLS = [f"{p}_{c}" for p in IMU_PREFIXES for c in IMU_COLUMNS]


# ── Download / extraction ────────────────────────────────────────────────────

def download_dataset_file(dataset_url, output_path=DOWNLOAD_FILE):
    if not dataset_url:
        raise ValueError("HMOG dataset_url is empty. Add a valid download URL in config.json.")
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"INFO: Using existing dataset archive: {output_path}")
        return
    subprocess.run(["wget", "-c", "-O", str(output_path), dataset_url], check=True)
    print("Download completed!")


def extract_zip(zipped_file, to_folder):
    if shutil.which("7z") is None:
        raise RuntimeError("7z is required. Install it with: sudo apt-get install -y p7zip-full")
    subprocess.run(["7z", "x", "-y", f"-o{to_folder}", str(zipped_file)], check=True)
    for file_path in Path(to_folder).rglob("*.zip"):
        subprocess.run(["7z", "x", "-y", f"-o{file_path.parent}", str(file_path)], check=True)


def extract(absolute_path):
    print("Extracting... 🚀")
    extract_zip(Path(absolute_path), DATASET_DIR)
    print(f"File is unzipped in {DATASET_DIR} folder ✅")
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

def keystroke_feature_extract(keystroke):
    if keystroke.isnull().values.any():
        print("WARNING: Original keystroke dataframe contains NaN")
        keystroke.fillna(0, inplace=True)

    keystroke["press_time"] = keystroke["press_time"].astype(int)
    n = len(keystroke)

    # Vectorised hold-latency
    keystroke["hl"]  = keystroke["release_time"] - keystroke["press_time"]
    keystroke["key"] = keystroke["key_code"]

    # Digraph timings (shifted by 1)
    keystroke["di_ud"] = keystroke["press_time"].shift(-1)   - keystroke["release_time"]
    keystroke["di_dd"] = keystroke["press_time"].shift(-1)   - keystroke["press_time"]
    keystroke["di_uu"] = keystroke["release_time"].shift(-1) - keystroke["release_time"]
    keystroke["di_du"] = keystroke["release_time"].shift(-1) - keystroke["press_time"]

    # Trigraph timings (shifted by 2)
    keystroke["tri_ud"] = keystroke["press_time"].shift(-2)   - keystroke["release_time"]
    keystroke["tri_dd"] = keystroke["press_time"].shift(-2)   - keystroke["press_time"]
    keystroke["tri_uu"] = keystroke["release_time"].shift(-2) - keystroke["release_time"]
    keystroke["tri_du"] = keystroke["release_time"].shift(-2) - keystroke["press_time"]

    # Last rows have no valid future context → zero-fill
    keystroke.iloc[-1, keystroke.columns.get_loc("di_ud"):] = 0
    keystroke.iloc[-2, keystroke.columns.get_loc("tri_ud"):] = 0
    keystroke.fillna(0, inplace=True)

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
            print(f"WARNING: {name} dataframe contains NaN")
            df.replace(np.nan, 0, inplace=True)

    def time_bounds(df):
        if df.empty:
            return math.inf, -math.inf
        return df.iloc[0]["event_time"], df.iloc[-1]["event_time"]

    acc_min, acc_max = time_bounds(acc)
    gyr_min, gyr_max = time_bounds(gyr)
    mag_min, mag_max = time_bounds(mag)

    start_time = min(acc_min, gyr_min, mag_min)
    highest_time = max(acc_max, gyr_max, mag_max)

    rows = []
    t = start_time
    while t < highest_time:
        t_end = t + sync_period
        slices = {
            "a": acc.loc[(acc["event_time"] >= t) & (acc["event_time"] <= t_end)],
            "g": gyr.loc[(gyr["event_time"] >= t) & (gyr["event_time"] <= t_end)],
            "m": mag.loc[(mag["event_time"] >= t) & (mag["event_time"] <= t_end)],
        }
        if any(s.empty for s in slices.values()):
            print("WARNING: Within sync period there are no elements")

        row = [
            (slices[p][c].mean() or 0.0)   # NaN → 0.0
            for p in IMU_PREFIXES
            for c in IMU_COLUMNS
        ]
        rows.append(row)
        t = t_end

    imu_sequence = pd.DataFrame(rows, columns=IMU_FULL_COLS)

    if len(imu_sequence) > imu_sequence_length:
        imu_sequence = imu_sequence.head(imu_sequence_length)
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


def read_keystroke(absolute_path, users_list):
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
            ks = keystroke_feature_extract(ks)

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
                [seq.drop(columns=["press_time", "release_time", "key_code"]).to_numpy(),
                 imu.to_numpy()]
                for seq, imu in zip(ks_seqs, imu_seqs)
            ]
            session_data.append(sequences)
            print(f"INFO: Session {session} completed")

        all_data.append(session_data)
        print(f"INFO: User {userid} completed ({i}/{len(users_list)})")

    return all_data


# ── Pickle helpers ───────────────────────────────────────────────────────────

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_data = Config().get_config_dict()["data"]

    dataset_url           = config_data["hmog"]["dataset_url"]
    keystroke_sequence_len = config_data["keystroke_sequence_len"] or 50
    imu_sequence_len       = config_data["imu_sequence_len"]       or 100
    windowing_offset       = config_data["hmog"]["windowing_offset"] or 5

    download_dataset_file(dataset_url)
    extract(DOWNLOAD_FILE)
    get_filtered_users()

    users_list = get_users(DATASET_DIR)
    train_users, val_test = train_test_split(users_list, test_size=30, train_size=69, shuffle=True)
    val_users, test_users = train_test_split(val_test,   test_size=15, train_size=15, shuffle=True)

    for split, users in [("training", train_users), ("validation", val_users), ("testing", test_users)]:
        save_pickle(read_keystroke(DATASET_DIR, users), f"{split}_keystroke_imu_data_all.pickle")