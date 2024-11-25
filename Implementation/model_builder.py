# model_builder.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_branches(genotype):
    """
    Build the branches of the model from the genotype.
    """
    gens = genotype.copy()
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    channels = []
    for element in gens:
        channels.append(element.pop(0))
    branches = [[], [], []]

    for i in range(len(gens)):
        for layer in gens[i]:
            if layer[0] == 'conv':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2D(channels[i][1], layer[1], **conv_args))
            elif layer[0] == 'dil_conv_d2':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2D(channels[i][1], layer[1], dilation_rate=2, **conv_args))
            elif layer[0] == 'dil_conv_d3':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2D(channels[i][1], layer[1], dilation_rate=3, **conv_args))
            elif layer[0] == 'dil_conv_d4':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2D(channels[i][1], layer[1], dilation_rate=4, **conv_args))
            elif layer[0] == 'Dsep_conv':
                for _ in range(layer[2]):
                    branches[i].extend([
                        layers.DepthwiseConv2D(layer[1], **conv_args),
                        layers.Conv2D(channels[i][1], 1, **conv_args)
                    ])
            elif layer[0] == 'invert_Bot_Conv_E2':
                expand = int(channels[i][1] * 2)
                for _ in range(layer[2]):
                    branches[i].extend([
                        layers.Conv2D(expand, 1, **conv_args),
                        layers.DepthwiseConv2D(layer[1], **conv_args),
                        layers.Conv2D(channels[i][1], 1, **conv_args)
                    ])
            elif layer[0] == 'conv_transpose':
                for _ in range(layer[2]):
                    branches[i].append(layers.Conv2DTranspose(channels[i][1], layer[1], **conv_args))
            elif layer[0] == 'identity':
                branches[i].append(layers.Lambda(lambda x: x))
            else:
                print("Unknown operation:", layer[0])
    bc = branches
    bc.append(channels[0][1])
    return bc

def get_model(genotype, upscale_factor=2, input_channels=3):
    """
    Build the Keras model from the genotype.
    """
    branch1, branch2, branch3, channels_mod = get_branches(genotype)

    conv_args = {
        "activation": "relu",
        "padding": "same",
    }

    inputs = layers.Input(shape=(None, None, input_channels))
    inp = layers.Conv2D(channels_mod, 3, **conv_args)(inputs)

    # Build branch 1
    b1 = inp
    for layer in branch1:
        b1 = layer(b1)

    # Build branch 2
    b2 = inp
    for layer in branch2:
        b2 = layer(b2)

    # Build branch 3
    b3 = inp
    for layer in branch3:
        b3 = layer(b3)

    x = layers.Add()([b1, b2, b3])
    x = layers.Conv2D(12, 3, **conv_args)(x)
    x = tf.nn.depth_to_space(x, upscale_factor)
    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    model = keras.Model(inputs, outputs)
    return model
