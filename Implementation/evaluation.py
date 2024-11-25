# evaluation.py

import tensorflow as tf
from tensorflow.keras import backend as K
from config import EVALUATION_METRIC, DATASET_TRAIN, DATASET_VAL, EPOCHS
from model_builder import get_model
from utils import psnr

def replace_activations_with_identity(model):
    """
    Replace all activation functions in the model with linear activations.

    Args:
        model (tf.keras.Model): The model whose activations will be replaced.
    """
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation is not None:
            layer.activation = tf.keras.activations.linear
        if isinstance(layer, tf.keras.Model):
            replace_activations_with_identity(layer)

def synflow_metric_nas(model):
    """
    Compute the SynFlow score for the given model.

    Args:
        model (tf.keras.Model): The model to evaluate.

    Returns:
        float: The SynFlow score.
    """
    replace_activations_with_identity(model)

    # Initialize weights and biases to 0.5
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.Constant(0.5)
        if hasattr(layer, 'bias_initializer'):
            layer.bias_initializer = tf.keras.initializers.Constant(0.5)
    model.build(input_shape=(None, 64, 64, 3))

    # Set weights and biases to 0.5
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.kernel.assign(tf.constant(0.5, shape=layer.kernel.shape))
        if hasattr(layer, 'bias'):
            layer.bias.assign(tf.constant(0.5, shape=layer.bias.shape))

    # Create a dummy input with all ones
    input_shape = (1, 64, 64, 3)
    data = tf.ones(input_shape)

    with tf.GradientTape() as tape:
        output = model(data)
        loss = tf.reduce_sum(output)

    # Compute gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # SynFlow score calculation
    synflow_score = 0.0
    for param, grad in zip(model.trainable_variables, grads):
        synflow_score += tf.reduce_sum(tf.abs(param * grad)).numpy()

    return synflow_score

def evaluate_model(individual):
    """
    Evaluate the model based on the selected evaluation metric (PSNR or SynFlow).

    Args:
        individual: The genotype representing the model architecture.

    Returns:
        float: The evaluation score (negative PSNR or negative SynFlow score).
    """
    genotype = individual  # Already decoded genotype
    model = get_model(genotype)

    if EVALUATION_METRIC == 'PSNR':
        # Training code with PSNR evaluation
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[psnr])

        # Check if datasets are defined
        if DATASET_TRAIN is None or DATASET_VAL is None:
            raise ValueError("DATASET_TRAIN and DATASET_VAL must be defined when using PSNR evaluation.")

        history = model.fit(DATASET_TRAIN, epochs=EPOCHS, validation_data=DATASET_VAL)
        valid_psnr = history.history['val_psnr'][-1]
        K.clear_session()
        return -valid_psnr  # Negative because we minimize in optimization

    elif EVALUATION_METRIC == 'SynFlow':
        synflow_score = synflow_metric_nas(model)
        K.clear_session()
        return -synflow_score  # Negative because we minimize

    else:
        raise ValueError("Invalid evaluation metric specified.")
        
