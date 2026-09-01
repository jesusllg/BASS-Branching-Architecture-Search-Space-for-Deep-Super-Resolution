import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bass import v2, v3
from bass.v3.evaluation import gradient_flow_diagnostics


def interaction_spec():
    skip = v3.BlockGene.skip()
    return v3.canonicalize_architecture(
        16,
        (
            (
                v3.BlockGene("attention", "channel_attention_residual", 0, 1),
                v3.BlockGene("cnn", "res_conv", 3, 1),
                v3.BlockGene("attention", "window_transformer", 4, 1),
            ),
            (
                v3.BlockGene("attention", "hybrid_conv_window", 4, 1),
                v3.BlockGene("cnn", "res_depthwise_separable", 5, 1),
                v3.BlockGene("attention", "regular_shifted_pair", 4, 1),
            ),
            (
                v3.BlockGene("cnn", "res_dilated_d2", 3, 1),
                v3.BlockGene("cnn", "inverted_residual_e2", 3, 1),
                skip,
            ),
        ),
        (v3.ExchangeGene.cimex(8), v3.ExchangeGene.cimex(16)),
    )


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_ibass_forward_backward_and_output_scale(scale):
    tf.keras.backend.clear_session()
    model = v3.build_model(
        interaction_spec(), upscale_factor=scale, input_shape=(8, 9, 3)
    )
    inputs = tf.random.uniform((1, 8, 9, 3), seed=7)
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = tf.reduce_mean(tf.square(outputs))
    gradients = tape.gradient(loss, model.trainable_variables)

    assert model.name == f"ibass_v3_x{scale}"
    assert tuple(outputs.shape) == (1, 8 * scale, 9 * scale, 3)
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(tf.reduce_all(tf.math.is_finite(item))) for item in gradients)


def test_ibass_feature_taps_keep_branchwise_v2_order():
    model, feature_model, names = v3.build_model(
        interaction_spec(),
        upscale_factor=2,
        input_shape=(8, 9, 3),
        return_feature_model=True,
    )
    del model
    features = feature_model(tf.zeros((1, 8, 9, 3)))
    assert len(features) == 9
    assert names[0] == "branch1.unit1"
    assert names[-1] == "branch3.unit3"
    assert all(tuple(feature.shape) == (1, 8, 9, 16) for feature in features)

    metadata = v3.feature_tap_metadata(interaction_spec())
    assert tuple(record["name"] for record in metadata) == names
    assert metadata[0]["tap_is_pre_exchange"] is True
    assert metadata[0]["exchange_after"] == "cimex_k8"
    assert metadata[2]["exchange_after"] is None
    assert any(record["internal_attention_blocks"] == 2 for record in metadata)


def test_ibass_accepts_dynamic_non_divisible_spatial_shapes():
    model = v3.build_model(interaction_spec(), upscale_factor=2)
    output = model(tf.random.uniform((1, 7, 11, 3), seed=37), training=False)
    assert tuple(output.shape) == (1, 14, 22, 3)


def test_v3_none_none_delegates_to_the_exact_v2_graph():
    previous = v2.sample(seed=31, attention_probability=0.5)
    extended = v3.migrate_v2(previous)
    sample = tf.random.uniform((1, 8, 9, 3), seed=29)

    tf.keras.utils.set_random_seed(101)
    expected_model = v2.build_model(previous, input_shape=(8, 9, 3))
    expected = expected_model(sample, training=False)
    expected_weights = [weight.numpy().copy() for weight in expected_model.weights]

    tf.keras.utils.set_random_seed(101)
    actual_model = v3.build_model(extended, input_shape=(8, 9, 3))
    actual = actual_model(sample, training=False)

    assert actual_model.name == "bass_v2_x2"
    assert actual_model.count_params() == expected_model.count_params()
    for actual_weight, expected_weight in zip(actual_model.weights, expected_weights):
        np.testing.assert_array_equal(actual_weight.numpy(), expected_weight)
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_ibass_gradient_proxy_sees_enabled_cimex_variables():
    model = v3.build_model(interaction_spec(), input_shape=(8, 9, 3))
    diagnostics = gradient_flow_diagnostics(model, input_shape=(8, 9, 3), strict=True)
    assert diagnostics.coverage == 1.0
    assert diagnostics.non_finite_variables == ()
    assert diagnostics.score > 0.0


def test_ibass_keras_save_load_round_trip(tmp_path):
    model = v3.build_model(interaction_spec(), input_shape=(8, 9, 3))
    sample = tf.random.uniform((1, 8, 9, 3), seed=3)
    expected = model(sample, training=False).numpy()
    path = tmp_path / "ibass_v3.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    actual = restored(sample, training=False).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
