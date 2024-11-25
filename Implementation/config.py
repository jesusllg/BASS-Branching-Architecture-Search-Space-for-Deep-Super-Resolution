# config.py

import numpy as np
import random
import tensorflow as tf

# Set random seeds for reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset Parameters
batch_size = 64
ratio = 2
patch_size = 64
overlap = 0.1

# Dataset Directories
directory1 = 'DIV2K_train_HR'
directory2 = 'DIV2K_valid_HR'

# Model Training Parameters
EPOCHS = 5
learning_rate = 3e-04
epsilon = 1e-07
weight_decay = 1e-8

# Evaluation Metric
EVALUATION_METRIC = 'SynFlow'  # Change to 'PSNR' if needed

# Device Configuration
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
