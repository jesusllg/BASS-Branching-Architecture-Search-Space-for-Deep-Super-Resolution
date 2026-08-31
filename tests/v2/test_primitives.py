import pytest

tf = pytest.importorskip("tensorflow")

from bass.v2.config import CHANNELS, PRIMITIVE_CONFIGS
from bass.v2.evaluation import gradient_flow_diagnostics
from bass.v2.genotype import BlockGene, canonicalize_architecture
from bass.v2.model_builder import build_model
from bass.v2.registry import make_unit_layers


def single_primitive_spec(channels, primitive, repeat=1):
    family, op, arg = primitive
    block = BlockGene(family, op, arg, repeat)
    skip = BlockGene.skip()
    return canonicalize_architecture(
        channels,
        ((block, skip, skip), (skip, skip, skip), (skip, skip, skip)),
    )


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("primitive", PRIMITIVE_CONFIGS)
def test_every_primitive_channel_pair_has_finite_forward_backward(channels, primitive):
    tf.keras.backend.clear_session()
    model = build_model(
        single_primitive_spec(channels, primitive),
        input_shape=(8, 8, 3),
        upscale_factor=2,
    )
    sample = tf.ones((1, 8, 8, 3))
    with tf.GradientTape() as tape:
        output = model(sample, training=True)
        loss = tf.reduce_mean(output)
    gradients = tape.gradient(loss, model.trainable_variables)
    assert tuple(output.shape) == (1, 16, 16, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(output)))
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(tf.reduce_all(tf.math.is_finite(item))) for item in gradients)


def test_max_repeat_attention_has_full_gradient_coverage():
    spec = single_primitive_spec(16, ("attention", "regular_shifted_pair", 4), repeat=3)
    model = build_model(spec, input_shape=(8, 8, 3), upscale_factor=2)
    diagnostics = gradient_flow_diagnostics(model, input_shape=(8, 8, 3), strict=True)
    assert diagnostics.coverage == 1.0
    assert diagnostics.non_finite_variables == ()


@pytest.mark.parametrize("primitive", PRIMITIVE_CONFIGS)
def test_every_primitive_has_an_identity_residual_skeleton(primitive):
    family, op, arg = primitive
    block = BlockGene(family, op, arg, 1)
    unit_layers = make_unit_layers(block, channels=16, name="residual_contract")
    inputs = tf.random.normal((1, 8, 8, 16), seed=13)
    outputs = inputs
    for layer in unit_layers:
        outputs = layer(outputs, training=False)
    for variable in [
        item for layer in unit_layers for item in layer.trainable_variables
    ]:
        variable.assign(tf.zeros_like(variable))
    outputs = inputs
    for layer in unit_layers:
        outputs = layer(outputs, training=False)
    tf.debugging.assert_near(outputs, inputs, atol=1e-6)
