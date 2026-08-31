import random

import pytest

from bass.config import V1_GENOME_BITS, V2_GENOME_BITS
from bass.encoding import (
    decode_v1_bits,
    gray_to_int,
    int_to_gray_bits,
)
from bass.v2 import (
    ArchitectureSpec,
    decode,
    decode_legacy_bits,
    encode,
    encode_legacy_bits,
    migrate_v1,
    repair_architecture,
    sample,
)


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


def test_v1_migration_enters_the_scientific_catalog_explicitly():
    rng = random.Random(73)
    old_bits = [rng.randint(0, 1) for _ in range(V1_GENOME_BITS)]
    upgraded = migrate_v1(old_bits)
    assert upgraded.schema_version == 2
    semantic = encode(upgraded)
    assert len(semantic) == 10
    assert decode(semantic) == upgraded


def test_legacy_93_bit_codec_is_import_export_only():
    legacy = decode_legacy_bits([0] * V2_GENOME_BITS)
    assert legacy.representation == "legacy93"
    assert encode_legacy_bits(legacy) == [0] * V2_GENOME_BITS


def test_v2_json_and_hash_are_deterministic():
    spec = sample(seed=42, attention_probability=0.5)
    restored = ArchitectureSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.canonical_json() == spec.canonical_json()
    assert restored.canonical_hash() == spec.canonical_hash()


def test_repair_is_idempotent():
    spec = sample(seed=11, attention_probability=0.75)
    repaired = repair_architecture(spec)
    assert repair_architecture(repaired) == repaired


def test_v2_repair_does_not_silently_absorb_v1():
    legacy = decode_v1_bits([0] * V1_GENOME_BITS)
    with pytest.raises(TypeError, match="bass.v2"):
        repair_architecture(legacy)


def test_sampling_can_create_free_hybrid_architectures():
    spec = sample(seed=9, attention_probability=0.5, skip_probability=0.0)
    assert 0.0 < spec.attention_fraction < 1.0


def test_legacy_codec_rejects_non_binary_chromosome():
    bits = [0] * V2_GENOME_BITS
    bits[3] = 2
    with pytest.raises(ValueError, match="binary"):
        decode_legacy_bits(bits)
