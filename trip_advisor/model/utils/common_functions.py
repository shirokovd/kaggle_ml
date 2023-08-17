import os

import numpy as np
import yaml


def load_config(config_file):
    with open(os.path.abspath(config_file)) as f:
        config = yaml.safe_load(f)
    return config


def one_hot(a):
    b = np.zeros((a.shape[0], a.max() + 1))
    b[np.arange(a.shape[0]), a] = 1
    return b
