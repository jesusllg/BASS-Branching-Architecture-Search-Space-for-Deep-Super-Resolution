# utils.py

import numpy as np
import tensorflow as tf

def soft_pareto_dominates(obj1, obj2):
    """
    Soft Pareto dominance: obj1 dominates obj2 if it's better in at least one objective and not worse in others.
    """
    better_in_any = False
    for a, b in zip(obj1, obj2):
        if a < b:
            better_in_any = True
        elif a > b:
            return False
    return better_in_any

def psnr(orig, pred):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between the original and predicted images.

    Args:
        orig (Tensor): Original images tensor.
        pred (Tensor): Predicted images tensor.

    Returns:
        Tensor: PSNR value.
    """
    # Scale and cast the target images to integer
    orig = tf.cast(orig * 255.0, tf.uint8)
    # Scale and cast the predicted images to integer
    pred = tf.cast(pred * 255.0, tf.uint8)
    # Return the PSNR
    return tf.image.psnr(orig, pred, max_val=255)

def Dominance(a_f, b_f):
    """
    Determine the Pareto dominance relationship between two solutions.

    Args:
        a_f: Objective values of solution a.
        b_f: Objective values of solution b.

    Returns:
        1 if a dominates b, -1 if b dominates a, 0 otherwise.
    """
    a_dominates = False
    b_dominates = False

    for a, b in zip(a_f, b_f):
        if a < b:
            a_dominates = True
        elif b < a:
            b_dominates = True

    if a_dominates and not b_dominates:
        return 1
    elif b_dominates and not a_dominates:
        return -1
    else:
        return 0
