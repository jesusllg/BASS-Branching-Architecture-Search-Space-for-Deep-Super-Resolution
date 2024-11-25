# evaluation.py

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from config import EVALUATION_METRIC, DATASET_TRAIN, DATASET_VAL, EPOCHS
from model_builder import get_model
from utils import psnr, soft_pareto_dominates
import numpy as np

def replace_activations_with_identity(model):
    """
    Replace all activation functions in the model with linear activations.
    """
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation is not None:
            layer.activation = tf.keras.activations.linear
        if isinstance(layer, tf.keras.Model):
            replace_activations_with_identity(layer)

def synflow_metric_nas(model):
    """
    Compute the SynFlow score for the given model with weights and biases initialized to 0.5.
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

    # SynFlow score
    synflow_score = 0.0
    for param, grad in zip(model.trainable_variables, grads):
        synflow_score += tf.reduce_sum(tf.abs(param * grad)).numpy()

    return synflow_score

def calculate_model_flops(model):
    """
    Calculate the FLOPs of the model.
    """
    # Ensure the model is built
    if not model.built:
        model.build(input_shape=(None, 64, 64, 3))

    # Convert Keras model to ConcreteFunction
    @tf.function
    def forward_pass(inputs):
        return model(inputs)

    concrete_func = forward_pass.get_concrete_function(tf.TensorSpec(shape=(1, 64, 64, 3), dtype=tf.float32))

    # Get frozen ConcreteFunction
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    # Profile the model
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts)

    # Get total FLOPs
    total_flops = flops.total_float_ops
    return total_flops

def count_params(model):
    """
    Count the total number of parameters in the model.
    """
    return model.count_params()

def evaluate_model(individual, n_eval):
    """
    Evaluate the model based on the selected evaluation metric (PSNR or SynFlow).
    
    Args:
        individual: The genotype representing the model architecture.
        n_eval: The evaluation number or iteration.

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
