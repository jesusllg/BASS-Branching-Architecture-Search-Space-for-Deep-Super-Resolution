# config.py

import numpy as np
import random
import tensorflow as tf

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Evaluation metric: 'PSNR' or 'SynFlow'
EVALUATION_METRIC = 'PSNR'  # Change to 'SynFlow' as needed

# Dataset Configuration
# Replace with your actual datasets
# Placeholder datasets (replace with actual data loaders)
train_images = np.random.rand(1000, 64, 64, 3).astype(np.float32)
train_labels = np.random.rand(1000, 64, 64, 3).astype(np.float32)
val_images = np.random.rand(200, 64, 64, 3).astype(np.float32)
val_labels = np.random.rand(200, 64, 64, 3).astype(np.float32)

DATASET_TRAIN = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
DATASET_VAL = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
EPOCHS = 5  # Number of epochs for training when using PSNR

# Device Configuration
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
