# model_builder.py

import tensorflow as tf
from tensorflow.keras import layers, models

def get_branches(genotype):
    """
    Build the branches of the model from the genotype.

    Args:
        genotype (Genotype): The genotype containing branches.

    Returns:
        tuple: Tuple containing the list of branches and the number of channels.
    """
    gens = [genotype.Branch1.copy(), genotype.Branch2.copy(), genotype.Branch3.copy()]
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    channels_list = []
    branches = [[], [], []]

    for i, branch in enumerate(gens):
        # Extract the channels information
        channels_info = branch.pop(0)
        channels_list.append(channels_info[1])

        # Build the layers for each branch
        for layer_info in branch:
            op_type, kernel_size, repeats = layer_info
            for _ in range(repeats):
                if op_type == 'conv':
                    branches[i].append(layers.Conv2D(channels_list[i], kernel_size, **conv_args))
                elif op_type == 'dil_conv_d2':
                    branches[i].append(layers.Conv2D(channels_list[i], kernel_size, dilation_rate=2, **conv_args))
                elif op_type == 'dil_conv_d3':
                    branches[i].append(layers.Conv2D(channels_list[i], kernel_size, dilation_rate=3, **conv_args))
                elif op_type == 'dil_conv_d4':
                    branches[i].append(layers.Conv2D(channels_list[i], kernel_size, dilation_rate=4, **conv_args))
                elif op_type == 'Dsep_conv':
                    branches[i].extend([
                        layers.DepthwiseConv2D(kernel_size, **conv_args),
                        layers.Conv2D(channels_list[i], 1, **conv_args)
                    ])
                elif op_type == 'invert_Bot_Conv_E2':
                    expand_channels = int(channels_list[i] * 2)
                    branches[i].extend([
                        layers.Conv2D(expand_channels, 1, **conv_args),
                        layers.DepthwiseConv2D(kernel_size, **conv_args),
                        layers.Conv2D(channels_list[i], 1, **conv_args)
                    ])
                elif op_type == 'conv_transpose':
                    branches[i].append(layers.Conv2DTranspose(channels_list[i], kernel_size, **conv_args))
                elif op_type == 'identity':
                    branches[i].append(layers.Lambda(lambda x: x))
                else:
                    raise ValueError(f"Unknown operation: {op_type}")

    return branches, channels_list[0]

def get_model(genotype, upscale_factor=2, input_channels=3):
    """
    Build the Keras model from the genotype.

    Args:
        genotype (Genotype): The genotype containing branches.
        upscale_factor (int): Upscaling factor for super-resolution.
        input_channels (int): Number of input channels.

    Returns:
        tf.keras.Model: The constructed Keras model.
    """
    branches, channels_mod = get_branches(genotype)
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }

    # Input layer
    inputs = layers.Input(shape=(None, None, input_channels))
    inp = layers.Conv2D(channels_mod, 3, **conv_args)(inputs)

    # Build branches
    b1 = inp
    for layer in branches[0]:
        b1 = layer(b1)

    b2 = inp
    for layer in branches[1]:
        b2 = layer(b2)

    b3 = inp
    for layer in branches[2]:
        b3 = layer(b3)

    # Merge branches
    x = layers.Add()([b1, b2, b3])

    # Reconstruction layers
    x = layers.Conv2D(12, 3, **conv_args)(x)
    x = tf.nn.depth_to_space(x, upscale_factor)
    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    # Build model
    model = models.Model(inputs, outputs)
    return model
