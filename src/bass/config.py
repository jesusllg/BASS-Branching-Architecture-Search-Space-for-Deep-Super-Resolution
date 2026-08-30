"""Stable search-space constants shared by both genotype versions."""

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

ATTENTION_PRIMITIVES = (
    "channel_attention",
    "window_attention",
    "shifted_window_attention",
    "hybrid_conv_attention",
)

CHANNELS = (16, 32, 48, 64)
KERNEL_SIZES = (1, 3, 5, 7)
REPEATS = (1, 2, 3, 4)
WINDOW_SIZES = (4, 8)

BRANCH_COUNT = 3
UNITS_PER_BRANCH = 3

V1_GENOME_BITS = 84
V1_DECODED_VALUES = 28
V2_GENOME_BITS = 93

HEADS_BY_CHANNELS = {
    16: 2,
    32: 4,
    48: 4,
    64: 8,
}

DEFAULT_UPSCALE = 2
DEFAULT_INPUT_CHANNELS = 3
DEFAULT_SEED = 42
