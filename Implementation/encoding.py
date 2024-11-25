# encoding.py

from collections import namedtuple

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

CHANNELS = [16, 32, 48, 64]
REPEAT = [1, 2, 3, 4]
KERNEL_SIZES = [1, 3, 5, 7]

Genotype = namedtuple('Genotype', 'Branch1 Branch2 Branch3')

def gray_to_int(gray_code):
    """
    Convert a Gray code string to an integer.

    Args:
        gray_code (str): Gray code string.

    Returns:
        int: Corresponding integer value.
    """
    binary_bits = [int(gray_code[0])]
    for i in range(1, len(gray_code)):
        next_bit = int(gray_code[i]) ^ binary_bits[i - 1]
        binary_bits.append(next_bit)
    binary_str = ''.join(str(bit) for bit in binary_bits)
    return int(binary_str, 2)

def bstr_to_rstr(bstring):
    """
    Convert a binary string to a list of integers by interpreting every 3 bits.

    Args:
        bstring (str): Binary string.

    Returns:
        list: List of integers representing operation indices.
    """
    rstr = []
    for i in range(0, len(bstring), 3):
        segment = bstring[i:i+3]
        if len(segment) < 3:
            segment = segment.ljust(3, '0')  # Pad with zeros if needed
        r = gray_to_int(segment)
        rstr.append(r)
    return rstr

def convert_cell(cell_bit_string):
    """
    Convert a cell bit-string to genome representation.

    Args:
        cell_bit_string (str): Bit-string representing a cell.

    Returns:
        list: Nested list representing the cell structure.
    """
    units = [cell_bit_string[i:i + 9] for i in range(0, len(cell_bit_string), 9)]
    cell = []
    for unit in units:
        unit_blocks = [unit[j:j + 3] for j in range(0, 9, 3)]
        cell.append(unit_blocks)
    return cell

def convert(bit_string):
    """
    Convert the network bit-string to genome representation for three branches.

    Args:
        bit_string (str): Full bit-string representing the network.

    Returns:
        list: List containing genome representations for each branch.
    """
    third = len(bit_string) // 3
    b1 = convert_cell(bit_string[:third])
    b2 = convert_cell(bit_string[third:2*third])
    b3 = convert_cell(bit_string[2*third:])
    return [b1, b2, b3]

def decode(genome):
    """
    Decode the genome into a Genotype with three branches.

    Args:
        genome (list): Genome list representing the network architecture.

    Returns:
        Genotype: Namedtuple containing branches with operations.
    """
    genome = genome.copy()
    channels_idx = genome.pop(0)
    channels_value = CHANNELS[channels_idx % len(CHANNELS)]
    genotype = convert(genome)
    branches = []

    for branch in genotype:
        branch_layers = [('channels', channels_value)]
        for block in branch:
            for unit in block:
                unit_values = bstr_to_rstr(''.join(unit))
                op_idx = unit_values[0] % len(PRIMITIVES)
                k_idx = unit_values[1] % len(KERNEL_SIZES)
                repeat_idx = unit_values[2] % len(REPEAT)
                branch_layers.append(
                    (PRIMITIVES[op_idx],
                     [KERNEL_SIZES[k_idx], KERNEL_SIZES[k_idx]],
                     REPEAT[repeat_idx])
                )
        branches.append(branch_layers)

    return Genotype(Branch1=branches[0], Branch2=branches[1], Branch3=branches[2])
