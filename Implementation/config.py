import numpy as np
import random
import tensorflow as tf

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Evaluation metric: 'PSNR' or 'SynFlow'
EVALUATION_METRIC = 'PSNR'  # Change to 'SynFlow' if needed

# NSGA-III Configuration
NSGA3_CONFIG = {
    'POP_SIZE': 100,       # Population size
    'N_GEN': 100,          # Number of generations
}

# Dataset Configuration
# Replace with your actual datasets
# Ensure that train_images, train_labels, val_images, and val_labels are defined
# For example:
# train_images = ...
# train_labels = ...
# val_images = ...
# val_labels = ...

# Uncomment and replace with actual data
# DATASET_TRAIN = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
# DATASET_VAL = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)

# For demonstration purposes, we'll leave these as placeholders
DATASET_TRAIN = None
DATASET_VAL = None

# Number of epochs for training when using PSNR
EPOCHS = 5

# Device Configuration
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
