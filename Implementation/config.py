"""Compatibility configuration for scripts written against BASS V1."""

import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bass.config import *

SEED = DEFAULT_SEED
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PRIMITIVES = list(CNN_PRIMITIVES)
REPEAT = list(REPEATS)
EVALUATION_METRIC = "SynFlow"
NSGA3_CONFIG = {"POP_SIZE": 20, "N_GEN": 10}
DATASET_TRAIN = None
DATASET_VAL = None
EPOCHS = 5
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
