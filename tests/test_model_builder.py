import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bass.model_builder import build_model
from bass.v2.genotype import ArchitectureSpec, BlockGene, architecture_from_blocks


def hybrid_spec() -> ArchitectureSpec:
    return architecture_from_blocks(
        16,
        (
            BlockGene("attention", "channel_attention_residual", 0, 1),
            BlockGene("attention", "window_transformer", 8, 1),
            BlockGene("attention", "regular_shifted_pair", 4, 1),
            BlockGene("attention", "hybrid_conv_window", 8, 2),
            BlockGene("cnn", "res_conv", 3, 1),
            BlockGene("cnn", "res_depthwise_separable", 5, 1),
            BlockGene("cnn", "res_dilated_d2", 3, 1),
            BlockGene("cnn", "inverted_residual_e2", 3, 1),
            BlockGene.skip(),
        ),
    )


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_hybrid_model_forward_backward_and_output_scale(scale):
    model = build_model(hybrid_spec(), upscale_factor=scale, input_shape=(16, 16, 3))
    inputs = tf.random.uniform((1, 16, 16, 3), seed=7)
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = tf.reduce_mean(outputs)
    gradients = tape.gradient(loss, model.trainable_variables)

    assert tuple(outputs.shape) == (1, 16 * scale, 16 * scale, 3)
    connected = [gradient for gradient in gradients if gradient is not None]
    assert connected
    assert all(bool(tf.reduce_all(tf.math.is_finite(item))) for item in connected)


def test_shifted_windows_accept_non_divisible_inference_shapes():
    model = build_model(hybrid_spec(), upscale_factor=2)
    outputs = model(tf.random.uniform((1, 13, 15, 3), seed=5), training=False)
    assert tuple(outputs.shape) == (1, 26, 30, 3)


def test_feature_taps_have_fixed_branchwise_cardinality():
    model, feature_model, names = build_model(
        hybrid_spec(),
        upscale_factor=2,
        input_shape=(16, 16, 3),
        return_feature_model=True,
    )
    del model
    features = feature_model(tf.zeros((1, 16, 16, 3)))
    assert len(features) == 9
    assert names[0] == "branch1.unit1"
    assert names[-1] == "branch3.unit3"
    assert all(tuple(feature.shape) == (1, 16, 16, 16) for feature in features)


def test_keras_save_load_round_trip(tmp_path):
    model = build_model(hybrid_spec(), upscale_factor=2, input_shape=(16, 16, 3))
    sample = tf.random.uniform((1, 16, 16, 3), seed=3)
    expected = model(sample, training=False).numpy()
    path = tmp_path / "hybrid_bass.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    actual = restored(sample, training=False).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_original_v1_zero_chromosome_builds():
    model = build_model([0] * 84, upscale_factor=2, input_shape=(12, 12, 3))
    output = model(tf.zeros((1, 12, 12, 3)))
    assert model.name == "bass_v1_x2"
    assert tuple(output.shape) == (1, 24, 24, 3)


def test_compatibility_builder_accepts_numpy_genomes():
    model = build_model(
        np.zeros(84, dtype=np.int8), upscale_factor=2, input_shape=(4, 4, 3)
    )
    assert model.name == "bass_v1_x2"


def test_builder_rejects_invalid_scale_and_shape():
    with pytest.raises(ValueError, match="upscale_factor"):
        build_model(hybrid_spec(), upscale_factor=5)
    with pytest.raises(ValueError, match="height, width, and channels"):
        build_model(hybrid_spec(), input_shape=(16, 3))


def test_v2_residual_head_has_an_exact_bicubic_zero_delta_baseline():
    model = build_model(
        hybrid_spec(), upscale_factor=2, input_shape=(8, 9, 3), head_mode="residual"
    )
    for variable in model.trainable_variables:
        variable.assign(tf.zeros_like(variable))
    inputs = tf.random.uniform((1, 8, 9, 3), seed=29)
    expected = tf.image.resize(inputs, (16, 18), method="bicubic")
    tf.debugging.assert_near(model(inputs, training=False), expected, atol=1e-6)


def test_v2_direct_head_remains_available_only_as_an_explicit_ablation():
    model = build_model(
        hybrid_spec(), upscale_factor=2, input_shape=(8, 9, 3), head_mode="direct"
    )
    outputs = model(tf.zeros((1, 8, 9, 3)), training=False)
    assert tuple(outputs.shape) == (1, 16, 18, 3)
