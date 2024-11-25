# encoding.py

from collections import namedtuple
import numpy as np

# Define the primitives and parameters
PRIMITIVES = [
    'conv',                # tf.keras.layers.Conv2D
    'dil_conv_d2',         # tf.keras.layers.Conv2D with dilation_rate=2
    'dil_conv_d3',         # tf.keras.layers.Conv2D with dilation_rate=3
    'dil_conv_d4',         # tf.keras.layers.Conv2D with dilation_rate=4
    'Dsep_conv',           # tf.keras.layers.DepthwiseConv2D
    'invert_Bot_Conv_E2',  # Inverted Bottleneck Block
    'conv_transpose',      # tf.keras.layers.Conv2DTranspose
    'identity'             # tf.keras.layers.Lambda (Identity)
]

CHANNELS = [16, 32, 48, 64, 16, 32, 48, 64]
REPEAT = [1, 2, 3, 4, 1, 2, 3, 4]
K = [1, 3, 5, 7, 1, 3, 5, 7]

Genotype = namedtuple('Genotype', 'Branch1 Branch2 Branch3')

def gray_to_int(gray_code):
    """
    Convert a Gray code string to an integer.
    """
    gray_bits = [int(bit) for bit in gray_code]
    binary_bits = [gray_bits[0]]
    for i in range(1, len(gray_bits)):
        next_bit = gray_bits[i] ^ binary_bits[i - 1]
        binary_bits.append(next_bit)
    binary_str = ''.join(str(bit) for bit in binary_bits)
    return int(binary_str, 2)

def bstr_to_rstr(bstring):
    """
    Convert a binary string to a list of integers by interpreting every 3 bits.
    """
    rstr = []
    for i in range(0, len(bstring), 3):
        r = gray_to_int(bstring[i:i+3])
        rstr.append(r)
    return rstr

def convert_cell(cell_bit_string):
    """
    Convert a cell bit-string to genome representation.
    """
    tmp = [cell_bit_string[i:i + 3] for i in range(0, len(cell_bit_string), 3)]
    return [tmp[i:i + 3] for i in range(0, len(tmp), 3)]

def convert(bit_string):
    """
    Convert the network bit-string to genome representation for three branches.
    """
    third = len(bit_string) // 3
    b1 = convert_cell(bit_string[:third])
    b2 = convert_cell(bit_string[third:2*third])
    b3 = convert_cell(bit_string[2*third:])
    return [b1, b2, b3]

def decode(genome):
    """
    Decode the genome into a Genotype with three branches.
    """
    genotype = genome.copy()
    channels = genome.pop(0)
    genotype = convert(genome)
    b1 = genotype[0]
    b2 = genotype[1]
    b3 = genotype[2]

    branch1 = [('channels', CHANNELS[channels])]
    branch2 = [('channels', CHANNELS[channels])]
    branch3 = [('channels', CHANNELS[channels])]

    for block in b1:
        for unit in block:
            unit = bstr_to_rstr(''.join(unit))
            branch1.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))

    for block in b2:
        for unit in block:
            unit = bstr_to_rstr(''.join(unit))
            branch2.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))

    for block in b3:
        for unit in block:
            unit = bstr_to_rstr(''.join(unit))
            branch3.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))

    return Genotype(Branch1=branch1, Branch2=branch2, Branch3=branch3)
