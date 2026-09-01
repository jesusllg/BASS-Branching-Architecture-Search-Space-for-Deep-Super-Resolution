import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bass.v3.blocks import CIMEXLayer


def test_cimex_preserves_shape_branch_sum_and_permutation_equivariance():
    layer = CIMEXLayer(channels=16, prototypes=8, gate_init=0.1)
    inputs = [tf.random.normal((2, 7, 9, 16), seed=seed) for seed in (3, 5, 7)]
    outputs = layer(inputs, training=False)

    assert len(outputs) == 3
    assert all(tuple(output.shape) == (2, 7, 9, 16) for output in outputs)
    corrections = tf.stack(
        [output - input_ for output, input_ in zip(outputs, inputs)], axis=1
    )
    tf.debugging.assert_near(
        tf.reduce_sum(corrections, axis=1),
        tf.zeros_like(corrections[:, 0]),
        atol=2e-6,
    )

    permutation = (2, 0, 1)
    permuted_outputs = layer([inputs[index] for index in permutation], training=False)
    for output, index in zip(permuted_outputs, permutation):
        tf.debugging.assert_near(output, outputs[index], atol=2e-6)


def test_cimex_has_finite_connected_and_nonzero_gradients():
    layer = CIMEXLayer(channels=16, prototypes=8)
    inputs = [tf.random.normal((1, 6, 5, 16), seed=seed) for seed in (11, 13, 17)]
    with tf.GradientTape() as tape:
        outputs = layer(inputs, training=True)
        loss = sum(
            (index + 1) * tf.reduce_mean(tf.square(output))
            for index, output in enumerate(outputs)
        )
    gradients = tape.gradient(loss, layer.trainable_variables)

    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(tf.reduce_all(tf.math.is_finite(item))) for item in gradients)
    assert all(bool(tf.reduce_any(item != 0)) for item in gradients)
    assert float(tf.math.tanh(layer.alpha).numpy()) == pytest.approx(0.01)


def test_cimex_config_round_trip_preserves_the_effective_gate():
    layer = CIMEXLayer(
        channels=32,
        prototypes=16,
        memory_dim=12,
        gate_init=0.05,
        name="exchange",
    )
    restored = CIMEXLayer.from_config(layer.get_config())
    assert restored.channels == 32
    assert restored.prototypes == 16
    assert restored.memory_dim == 12
    assert restored.gate_init == pytest.approx(0.05)


def test_cimex_rejects_invalid_configuration_and_shapes():
    with pytest.raises(ValueError, match="prototypes"):
        CIMEXLayer(channels=16, prototypes=0)
    with pytest.raises(ValueError, match="gate_init"):
        CIMEXLayer(channels=16, prototypes=8, gate_init=1.0)

    layer = CIMEXLayer(channels=16, prototypes=8)
    bad = [np.zeros((1, 4, 4, 16), dtype=np.float32)] * 2
    with pytest.raises(ValueError, match="exactly 3"):
        layer(bad)


def test_cimex_supports_mixed_float16_with_float32_gate_state():
    previous_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        layer = CIMEXLayer(channels=16, prototypes=8)
        inputs = [
            tf.random.normal((1, 5, 7, 16), seed=seed, dtype=tf.float16)
            for seed in (19, 23, 29)
        ]
        with tf.GradientTape() as tape:
            outputs = layer(inputs, training=True)
            loss = sum(tf.reduce_mean(tf.square(output)) for output in outputs)
        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(output.dtype == tf.float16 for output in outputs)
        assert layer.alpha.dtype == tf.float32
        assert all(gradient is not None for gradient in gradients)
        assert all(
            bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients
        )
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)
        tf.keras.backend.clear_session()
