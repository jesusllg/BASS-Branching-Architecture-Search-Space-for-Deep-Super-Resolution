"""Frozen constants for the CNN-only BASS V1 search space."""

CNN_PRIMITIVES = (
    "conv",
    "dil_conv_d2",
    "dil_conv_d3",
    "dil_conv_d4",
    "depthwise_separable_conv",
    "inverted_bottleneck_e2",
    "conv_transpose",
    "identity",
)

CHANNELS = (16, 32, 48, 64)
KERNEL_SIZES = (1, 3, 5, 7)
REPEATS = (1, 2, 3, 4)

BRANCH_COUNT = 3
UNITS_PER_BRANCH = 3
GENOME_BITS = 84
DECODED_VALUES = 28

DEFAULT_UPSCALE = 2
DEFAULT_INPUT_CHANNELS = 3
DEFAULT_SEED = 42
