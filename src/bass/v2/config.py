"""Frozen constants for the publication-oriented hybrid BASS V2 space."""

CHANNELS = (16, 32, 48, 64)
REPEATS = (1, 2, 3)
WINDOW_SIZES = (4, 8)

BRANCH_COUNT = 3
UNITS_PER_BRANCH = 3
SEMANTIC_GENOME_LENGTH = 1 + BRANCH_COUNT * UNITS_PER_BRANCH
LEGACY_GENOME_BITS = 93

# Each entry includes every meaningful architectural argument. Scientific-search
# genes therefore never carry an inactive kernel or window field.
CNN_PRIMITIVE_CONFIGS = (
    ("res_conv", 3),
    ("res_conv", 5),
    ("res_dilated_d2", 3),
    ("res_depthwise_separable", 3),
    ("res_depthwise_separable", 5),
    ("inverted_residual_e2", 3),
    ("inverted_residual_e2", 5),
)

ATTENTION_PRIMITIVE_CONFIGS = (
    ("channel_attention_residual", 0),
    ("window_transformer", 4),
    ("window_transformer", 8),
    ("regular_shifted_pair", 4),
    ("regular_shifted_pair", 8),
    ("hybrid_conv_window", 4),
    ("hybrid_conv_window", 8),
)

PRIMITIVE_CONFIGS = tuple(
    [("cnn", op, arg) for op, arg in CNN_PRIMITIVE_CONFIGS]
    + [("attention", op, arg) for op, arg in ATTENTION_PRIMITIVE_CONFIGS]
)

CNN_PRIMITIVES = tuple(dict.fromkeys(op for op, _ in CNN_PRIMITIVE_CONFIGS))
ATTENTION_PRIMITIVES = tuple(dict.fromkeys(op for op, _ in ATTENTION_PRIMITIVE_CONFIGS))
KERNEL_SIZES = tuple(sorted({arg for _, arg in CNN_PRIMITIVE_CONFIGS if arg > 0}))

# State 0 is skip. Remaining states are primitive-major, then repeat-major.
UNIT_STATE_COUNT = 1 + len(PRIMITIVE_CONFIGS) * len(REPEATS)

HEADS_BY_CHANNELS = {
    16: 2,
    32: 4,
    48: 4,
    64: 8,
}

DEFAULT_UPSCALE = 2
DEFAULT_INPUT_CHANNELS = 3
DEFAULT_SEED = 42
DEFAULT_HEAD_MODE = "residual"
RESIDUAL_SCALE = 0.1
