from pathlib import Path
import shutil
import sys
sys.path.append(str((Path(__file__)/"../../../utils").resolve()))

import subprocess
import pandas as pd
from Config import Config

dataset_url = Config().get_config_dict()["data"]["aalto"]["dataset_url"]

if shutil.which("7z") is None:
    raise RuntimeError("7z is required. Install it with: sudo apt-get install -y p7zip-full")

subprocess.run(["wget", "-c", dataset_url], check=True)
subprocess.run(["7z", "x", "-y", "csv_raw_and_processed.zip"], check=True)

headers = pd.read_csv("Data_Raw/test_sections_header.csv").Field.tolist()
test_sections = pd.read_csv("Data_Raw/test_sections.csv", names=headers, encoding='latin-1', on_bad_lines='skip')

test_sections = test_sections[['TEST_SECTION_ID', 'PARTICIPANT_ID']]

headers = pd.read_csv("Data_Raw/keystrokes_header.csv").Field.tolist()
data = pd.read_csv("Data_Raw/keystrokes.csv", names=headers, encoding='latin-1', chunksize=10000000, on_bad_lines='skip')
dfs = []
for df in data:
    dfs.append(df[["TEST_SECTION_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE"]].merge(test_sections))

keystroke = pd.concat(dfs)

keystroke.rename(columns = {'TEST_SECTION_ID':'session_id', 'PARTICIPANT_ID': 'user_id', 'PRESS_TIME':'press_time', 'RELEASE_TIME':'release_time', 'KEYCODE':'key_code'}, inplace = True)

keystroke.to_csv("data/AaltoDB/prep_data/keystroke_data.csv")
