import random

import pytest

from bass.config import V1_GENOME_BITS, V2_GENOME_BITS
from bass.encoding import (
    decode_v1_bits,
    decode_v2_bits,
    encode_v2_bits,
    gray_to_int,
    int_to_gray_bits,
    sample_v2,
    upgrade_v1,
)
from bass.genotype import ArchitectureSpec
from bass.repair import repair_architecture


def test_gray_code_round_trip():
    for value in range(16):
        assert gray_to_int(int_to_gray_bits(value, 4)) == value


def test_zero_v1_chromosome_decodes_original_space():
    spec = decode_v1_bits([0] * V1_GENOME_BITS)
    assert spec.schema_version == 1
    assert spec.channels == 16
    assert len(spec.branches) == 3
    assert all(len(branch) == 3 for branch in spec.branches)
    assert all(block.family == "cnn" for block in spec.flat_blocks)
    assert all(block.op == "conv" for block in spec.flat_blocks)
    assert all(block.arg == 1 for block in spec.flat_blocks)
    assert all(block.repeat == 1 for block in spec.flat_blocks)


def test_v1_upgrade_and_v2_codec_preserve_phenotype():
    rng = random.Random(73)
    old_bits = [rng.randint(0, 1) for _ in range(V1_GENOME_BITS)]
    upgraded = upgrade_v1(old_bits)
    assert upgraded.schema_version == 2
    v2_bits = encode_v2_bits(upgraded)
    assert len(v2_bits) == V2_GENOME_BITS
    assert decode_v2_bits(v2_bits) == upgraded


def test_v2_json_and_hash_are_deterministic():
    spec = sample_v2(seed=42, attention_probability=0.5)
    restored = ArchitectureSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.canonical_json() == spec.canonical_json()
    assert restored.canonical_hash() == spec.canonical_hash()


def test_repair_is_idempotent():
    spec = sample_v2(seed=11, attention_probability=0.75)
    repaired = repair_architecture(spec)
    assert repair_architecture(repaired) == repaired


def test_repair_preserves_schema_version():
    legacy = decode_v1_bits([0] * V1_GENOME_BITS)
    assert repair_architecture(legacy).schema_version == 1


def test_sampling_can_create_free_hybrid_architectures():
    spec = sample_v2(seed=9, attention_probability=0.5)
    assert 0.0 < spec.attention_fraction < 1.0


def test_v2_rejects_non_binary_chromosome():
    bits = [0] * V2_GENOME_BITS
    bits[3] = 2
    with pytest.raises(ValueError, match="binary"):
        decode_v2_bits(bits)
