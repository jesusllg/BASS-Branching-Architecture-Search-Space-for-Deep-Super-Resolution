import pytest

from bass import v2, v3
from bass.v3.genotype import ArchitectureSpec, ExchangeGene


def test_v3_has_a_strict_12_integer_scientific_codec():
    genome = v3.sample_genome(seed=8)
    assert len(genome) == 12
    assert v3.decode(genome).schema_version == 3
    with pytest.raises(ValueError, match="12"):
        v3.decode([0] * 10)
    with pytest.raises(TypeError, match="integer"):
        v3.decode([0.0] * 12)
    with pytest.raises(TypeError, match="integer"):
        v3.decode([False] * 12)


def test_v3_direct_sampler_is_canonical_and_reproducible():
    first = v3.sample_canonical_genome(seed=73)
    second = v3.sample_canonical_genome(seed=73)
    assert first == second
    assert v3.encode(v3.decode(first)) == first
    assert v3.canonicalize_genome(first) == first


def test_cimex_is_optional_at_each_exchange_site():
    no_exchange = v3.sample(seed=4, exchange_probability=0.0)
    all_exchange = v3.sample(seed=4, exchange_probability=1.0)

    assert no_exchange.exchange_count == 0
    assert all_exchange.exchange_count == 2
    assert all(item.op == "none" for item in no_exchange.exchanges)
    assert all(item.op == "cimex" for item in all_exchange.exchanges)


def test_v2_embedding_is_exact_and_explicit():
    previous = v2.sample(seed=17, attention_probability=0.5)
    extended = v3.migrate_v2(previous)

    assert extended.channels == previous.channels
    assert extended.branches == previous.branches
    assert extended.exchanges == (ExchangeGene.none(), ExchangeGene.none())
    assert v3.to_v2(extended) == previous
    assert v3.decode(v3.encode(extended)) == extended


def test_enabled_exchange_cannot_be_silently_projected_to_v2():
    spec = v3.sample(seed=2, exchange_probability=1.0)
    with pytest.raises(ValueError, match="enabled CIMEX"):
        v3.to_v2(spec)


def test_branch_permutations_share_one_v3_canonical_hash():
    a = v3.BlockGene("cnn", "res_conv", 3, 1)
    b = v3.BlockGene("attention", "window_transformer", 4, 1)
    skip = v3.BlockGene.skip()
    exchanges = (ExchangeGene.cimex(8), ExchangeGene.none())
    branches = ((a, b, skip), (skip, a, b), (skip, skip, skip))
    left = v3.canonicalize_architecture(
        16,
        branches,
        exchanges,
    )
    right = v3.canonicalize_architecture(
        16,
        (branches[2], branches[0], branches[1]),
        exchanges,
    )
    assert left == right
    assert left.canonical_hash() == right.canonical_hash()


def test_v3_json_and_hash_are_deterministic():
    spec = v3.sample(seed=42, exchange_probability=1.0)
    restored = ArchitectureSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.canonical_json() == spec.canonical_json()
    assert restored.canonical_hash() == spec.canonical_hash()


def test_stage_unsafe_v1_representation_is_rejected_explicitly():
    spec = v3.sample(seed=42, exchange_probability=1.0)
    payload = spec.to_dict()
    payload["representation"] = "interaction-semantic-v1"

    with pytest.raises(ValueError, match="stage-unsafe quotient"):
        ArchitectureSpec.from_dict(payload)


def test_exchange_gene_validation_is_strict():
    with pytest.raises(ValueError, match="Unsupported"):
        ExchangeGene("cimex", 32)
    with pytest.raises(TypeError, match="integer"):
        ExchangeGene("cimex", True)
    with pytest.raises(ValueError, match="fields"):
        ExchangeGene.from_dict({"op": "none", "prototypes": 0, "extra": 1})


def test_algebraically_inactive_centered_exchanges_are_canonicalized_away():
    active = v3.BlockGene("cnn", "res_conv", 3, 1)
    downstream = v3.BlockGene("cnn", "res_depthwise_separable", 3, 1)
    skip = v3.BlockGene.skip()
    spec = v3.canonicalize_architecture(
        16,
        (
            (active, downstream, skip),
            (active, downstream, skip),
            (active, downstream, skip),
        ),
        (ExchangeGene.cimex(8), ExchangeGene.cimex(16)),
    )
    assert spec.exchanges == (ExchangeGene.cimex(8), ExchangeGene.none())

    fully_inactive = v3.canonicalize_architecture(
        16,
        ((active, skip, skip), (active, skip, skip), (active, skip, skip)),
        (ExchangeGene.cimex(8), ExchangeGene.cimex(16)),
    )
    assert fully_inactive.exchanges == (ExchangeGene.none(), ExchangeGene.none())


def test_enabled_exchange_is_a_hard_internal_skip_barrier():
    a = v3.BlockGene("cnn", "res_conv", 3, 1)
    b = v3.BlockGene("cnn", "res_conv", 5, 1)
    c = v3.BlockGene("cnn", "res_dilated_d2", 3, 1)
    skip = v3.BlockGene.skip()

    spec = v3.canonicalize_architecture(
        16,
        ((skip, a, skip), (b, skip, skip), (c, skip, skip)),
        (ExchangeGene.cimex(8), ExchangeGene.none()),
    )

    branch = next(item for item in spec.branches if a in item)
    assert branch == (skip, a, skip)
    assert spec.exchanges == (ExchangeGene.cimex(8), ExchangeGene.none())


def test_repeat_runs_do_not_merge_across_enabled_exchange():
    a1 = v3.BlockGene("cnn", "res_conv", 3, 1)
    a2 = v3.BlockGene("cnn", "res_conv", 3, 2)
    b = v3.BlockGene("cnn", "res_conv", 5, 1)
    skip = v3.BlockGene.skip()

    spec = v3.canonicalize_architecture(
        16,
        ((a1, a2, skip), (b, skip, skip), (skip, skip, skip)),
        (ExchangeGene.cimex(16), ExchangeGene.none()),
    )

    branch = next(item for item in spec.branches if a1 in item)
    assert branch == (a1, a2, skip)
    assert spec.exchanges[0] == ExchangeGene.cimex(16)


def test_none_exchange_allows_safe_skip_and_repeat_compression():
    a1 = v3.BlockGene("cnn", "res_conv", 3, 1)
    a2 = v3.BlockGene("cnn", "res_conv", 3, 2)
    skip = v3.BlockGene.skip()

    spec = v3.canonicalize_architecture(
        16,
        ((skip, a1, a2), (skip, skip, skip), (skip, skip, skip)),
        (ExchangeGene.none(), ExchangeGene.none()),
    )

    compact = v3.BlockGene("cnn", "res_conv", 3, 3)
    assert (compact, skip, skip) in spec.branches


def test_corrected_stage_aware_space_has_reproducible_exact_cardinality():
    assert len(v3.canonical_branch_genomes((False, False))) == 68_923
    assert len(v3.canonical_branch_genomes((True, False))) == 74_089
    assert len(v3.canonical_branch_genomes((False, True))) == 74_089
    assert len(v3.canonical_branch_genomes((True, True))) == 79_507
    assert v3.canonical_architecture_count() == 2_643_101_795_040_984

    counts = v3.canonical_exchange_configuration_counts()
    assert len(counts) == 9
    assert sum(counts.values()) == v3.canonical_architecture_count()
