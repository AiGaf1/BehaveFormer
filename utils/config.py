import json
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch


class Config:
    _instance = None
    _data = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path = "../../config.json"):
        self.path = (Path(__file__)/path).resolve()

    def get_config_dict(self):
        if not Config._data:
            with open(self.path) as file:
                Config._data = json.load(file)

        return Config._data


def seed_training(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
