# BASS: Branching Architecture Search Space for Super-Resolution

## Introduction

The objective of this repository is to provide a platform for users to utilize our search space, known as **BASS** (Branching Architecture Search Space), dedicated to the automatic design of neural network architectures for super-resolution image restoration. We employ **Multi-Depth Branch Networks** as the foundation for this search space over which a multi-objective optimization process is used to design effective architectures.

This project focuses on discovering optimal neural architectures that balance multiple objectives:

- **Performance**: Measured by PSNR (Peak Signal-to-Noise Ratio) or SynFlow.
- **Model Complexity**: The number of parameters in the model.
- **Computational Cost**: Measured in FLOPs (Floating Point Operations).

## Features

- **Branching Architecture Search Space (BASS)**: A flexible framework for defining complex neural network architectures.
- **NSGA-III Algorithm**: Utilizes NSGA-III for multi-objective optimization to find Pareto-optimal architectures.
- **Multi-Objective Evaluation**: Supports evaluation based on PSNR, SynFlow, parameter count, and FLOPs.
- **Dynamic Model Construction**: Automatically builds Keras models from encoded genotypes.
- **Reproducibility**: Configurable random seeds for consistent results.

## How It Works

### Encoding and Decoding

- **Genome Representation**: Architectures are represented as bitstrings (genomes), encoding the choices of operations, kernel sizes, channels, and repeats.
- **Encoding**: The `encoding.py` module handles the conversion of bitstrings into structured genotypes, which define the architecture of each branch.
- **Decoding**: Genotypes are decoded to build actual TensorFlow/Keras models using the `model_builder.py` module.

### Model Building

- **Branching Structure**: Each model consists of three branches, each defined by its genotype.
- **Layer Primitives**: The `PRIMITIVES` list defines available operations, such as convolutions, dilated convolutions, depthwise separable convolutions, and more.
- **Dynamic Construction**: The `get_model` function constructs the model by iterating over the genotype and adding layers accordingly.

### Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of reconstructed images in super-resolution tasks.
- **SynFlow**: A proxy metric for evaluating neural network architectures without full training, focusing on network connectivity.
- **FLOPs and Parameters**: Computational cost is measured using FLOPs, and model complexity is assessed by counting the number of parameters.

## Project Structure

```
BASS/
├── config.py
├── encoding.py
├── evaluation.py
├── model_builder.py
├── nsga3.py
├── utils.py
├── main.py
├── README.md
├── LICENSE
└── requirements.txt
```

- **config.py**: Configuration settings for the project, including dataset paths and hyperparameters.
- **encoding.py**: Functions for encoding and decoding genome representations.
- **evaluation.py**: Evaluation functions for PSNR, SynFlow, FLOPs, and parameter counting.
- **model_builder.py**: Functions to dynamically build Keras models from genotypes.
- **nsga3.py**: Implementation of the NSGA-III algorithm.
- **utils.py**: Utility functions, such as dominance checks.
- **main.py**: Main script to run the NSGA-III optimization process.
- **README.md**: This document.
- **LICENSE**: License information.
- **requirements.txt**: List of dependencies.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/BASS.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd BASS
   ```

3. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have TensorFlow, NumPy, and other necessary libraries installed.

## Usage

### Configuration

Before running the algorithm, configure the settings in `config.py`:

- **Random Seed**: Set `SEED` for reproducibility.
- **Dataset Paths**: Update `DATASET_TRAIN` and `DATASET_VAL` with your dataset paths.
- **Model Parameters**: Adjust `EPOCHS`, `learning_rate`, etc., as needed.
- **Evaluation Metric**: Set `EVALUATION_METRIC` to `'PSNR'` or `'SynFlow'`.

```python
# config.py

import numpy as np
import random
import tensorflow as tf

# Set random seeds for reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Evaluation Metric
EVALUATION_METRIC = 'SynFlow'  # Change to 'PSNR' if needed

# NSGA-III Configuration
NSGA3_CONFIG = {
    'POP_SIZE': 100,       # Population size
    'N_GEN': 100,          # Number of generations
}

# Dataset Configuration
# Replace with your actual datasets
DATASET_TRAIN = None  # Replace with your training dataset
DATASET_VAL = None    # Replace with your validation dataset

EPOCHS = 5  # Number of epochs for training when using PSNR

# Device Configuration
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
```

### Running the Algorithm

Use `main.py` to execute the NSGA-III optimization process.

```bash
python main.py
```

After running, the script will output the final population and non-dominated solutions based on the defined objectives.

## Algorithm Implemented

### NSGA-III

An evolutionary algorithm designed for solving complex multi-objective optimization problems. NSGA-III uses reference points to maintain diversity and spread among the solutions.

- **Initialization**: Generates an initial population randomly.
- **Selection**: Uses tournament selection based on Pareto dominance.
- **Crossover and Mutation**: Applies k-point crossover and bit-flip mutation to generate offspring.
- **Non-Dominated Sorting**: Ranks solutions into different fronts based on Pareto dominance.
- **Reference Points**: Uses predefined reference points to guide the selection process for the next generation.

## Research Publications

This project is part of ongoing research in neural architecture search and multi-objective optimization. It contributes to the field by integrating a flexible search space with advanced optimization algorithms.

**Note**: Please add any relevant publications or research articles if available.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bug fix.
3. **Commit your changes** with clear messages.
4. **Push to your fork**.
5. **Submit a pull request**.

Please ensure your code adheres to the project's coding standards and includes proper documentation.

Ensure the following dependencies are installed before running the project:

**requirements.txt**

```
numpy
tensorflow
tqdm
```

**Note**: Depending on your system and specific versions of TensorFlow and other packages, you may need to specify version numbers. For example:

```
numpy>=1.18.0
tensorflow>=2.4.0
tqdm>=4.50.0
```

To install the requirements, run:

```bash
pip install -r requirements.txt
```

---

Feel free to reach out if you have any questions or need assistance setting up the project.
