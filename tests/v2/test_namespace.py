import pytest

from bass import v1, v2
from bass.v2.genotype import ArchitectureSpec, BlockGene


def test_attention_is_optional_per_unit_in_v2():
    cnn_only = v2.sample(seed=4, attention_probability=0.0, skip_probability=0.0)
    attention_only = v2.sample(seed=4, attention_probability=1.0, skip_probability=0.0)
    mixed = v2.sample(seed=9, attention_probability=0.5, skip_probability=0.0)

    assert cnn_only.attention_fraction == 0.0
    assert attention_only.attention_fraction == 1.0
    assert 0.0 < mixed.attention_fraction < 1.0


def test_v2_has_a_strict_10_integer_scientific_codec():
    genome = v2.sample_genome(seed=8)
    assert len(genome) == 10
    assert v2.decode(genome).schema_version == 2
    with pytest.raises(ValueError, match="10"):
        v2.decode([0] * 93)
    with pytest.raises(TypeError, match="integer"):
        v2.decode([0.0] * 10)
    with pytest.raises(TypeError, match="integer"):
        v2.decode([False] * 10)


def test_v2_rejects_the_unimplemented_synflow_label_before_search():
    with pytest.raises(ValueError, match="does not implement canonical SynFlow"):
        v2.BASSProblem(metric="synflow")


def test_explicit_v1_migration_is_cnn_only_but_not_claimed_exact():
    original = v1.decode([0] * 84)
    upgraded = v2.migrate_v1(original)

    assert upgraded.schema_version == 2
    assert upgraded.channels == original.channels
    assert upgraded.attention_fraction == 0.0
    assert all(block.family in {"cnn", "skip"} for block in upgraded.flat_blocks)
    assert v2.decode(v2.encode(upgraded)) == upgraded


def test_branch_permutations_and_skip_positions_share_one_canonical_hash():
    a = BlockGene("cnn", "res_conv", 3, 1)
    b = BlockGene("attention", "window_transformer", 4, 1)
    skip = BlockGene.skip()
    left = v2.canonicalize_architecture(
        16,
        ((a, skip, b), (skip, skip, skip), (b, a, skip)),
    )
    right = v2.canonicalize_architecture(
        16,
        ((b, a, skip), (skip, a, b), (skip, skip, skip)),
    )
    assert left == right
    assert left.canonical_hash() == right.canonical_hash()


def test_equivalent_repeat_groupings_are_canonicalized():
    a1 = BlockGene("cnn", "res_conv", 3, 1)
    a2 = BlockGene("cnn", "res_conv", 3, 2)
    a3 = BlockGene("cnn", "res_conv", 3, 3)
    skip = BlockGene.skip()
    grouped = v2.canonicalize_architecture(
        16, ((a1, a2, skip), (skip, skip, skip), (skip, skip, skip))
    )
    compact = v2.canonicalize_architecture(
        16, ((a3, skip, skip), (skip, skip, skip), (skip, skip, skip))
    )
    assert grouped == compact


def test_persisted_architecture_validation_is_strict():
    a = BlockGene("cnn", "res_conv", 3, 1)
    skip = BlockGene.skip()
    with pytest.raises(ValueError, match="not canonical"):
        ArchitectureSpec(
            channels=16,
            branches=((skip, a, skip), (skip, skip, skip), (skip, skip, skip)),
        )

    payload = v2.sample(seed=5).to_dict()
    payload["branches"][0][0]["repeat"] = "1"
    with pytest.raises(TypeError, match="integers"):
        ArchitectureSpec.from_dict(payload)

    payload = v2.sample(seed=6).to_dict()
    del payload["representation"]
    with pytest.raises(ValueError, match="fields"):
        ArchitectureSpec.from_dict(payload)
