# utils.py

import tensorflow as tf

def psnr(y_true, y_pred):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between the true and predicted images.

    Args:
        y_true (Tensor): Original images tensor.
        y_pred (Tensor): Predicted images tensor.

    Returns:
        Tensor: PSNR value.
    """
    # Scale and cast the images to uint8
    y_true = tf.cast(y_true * 255.0, tf.uint8)
    y_pred = tf.cast(y_pred * 255.0, tf.uint8)
    # Return the PSNR
    return tf.image.psnr(y_true, y_pred, max_val=255)

def Dominance(a_f, b_f):
    """
    Determine the Pareto dominance relationship between two solutions.

    Args:
        a_f (list): Objective values of solution a.
        b_f (list): Objective values of solution b.

    Returns:
        int: 1 if a dominates b, -1 if b dominates a, 0 otherwise.
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
        
