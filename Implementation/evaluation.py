# evaluation.py

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import os
import contextlib
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from config import EVALUATION_METRIC, DATASET_TRAIN, DATASET_VAL, EPOCHS, DEVICE
from model_builder import get_model
from utils import psnr

def replace_activations_with_identity(model):
    """
    Replace all activation functions in the model with linear activations.
    """
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation is not None:
            layer.activation = tf.keras.activations.linear
        if isinstance(layer, tf.keras.Model):
            replace_activations_with_identity(layer)

def compute_synflow_scores(model, input_size):
    input_tensor = tf.ones(input_size)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = model(input_tensor)
        objective = tf.ones_like(output)
    gradients = tape.gradient(output, model.trainable_weights, output_gradients=objective)
    scores = [tf.reduce_sum(tf.abs(w * g)) for w, g in zip(model.trainable_weights, gradients)]
    total_score = tf.reduce_sum(scores)
    return total_score.numpy()

def evaluate_network_with_synflow(model, input_size):
    synflow_score = compute_synflow_scores(model, input_size)
    return synflow_score

def calculate_model_flops(model):
    # Ensure the model is built with a defined input shape
    if not model.built:
        sample_input = tf.keras.Input(shape=model.input_shape[1:])
        model(sample_input)

    # Define the forward pass for the model
    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    # TensorFlow Profiler gets FLOPs
    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())

    # Divide by 2 as `profile` counts multiply and accumulate as two FLOPs
    flops = graph_info.total_float_ops // 2
    return flops

def count_params(model):
    """
    Count the total number of parameters in the model.
    """
    return model.count_params()

def psnr(orig, pred):
    # Scale and cast the target images to integer
    orig = tf.cast(orig * 255.0, tf.uint8)
    # Scale and cast the predicted images to integer
    pred = tf.cast(pred * 255.0, tf.uint8)
    # Return the PSNR
    return tf.image.psnr(orig, pred, max_val=255)
