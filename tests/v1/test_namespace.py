import json
import subprocess
import sys

import pytest

from bass import v1


def test_v1_import_does_not_load_v2_or_attention_modules():
    code = """
import json
import sys
import bass.v1
blocked = [
    name for name in sys.modules
    if name.startswith('bass.v2') or name == 'bass.blocks.attention'
]
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


def test_v1_has_a_strict_84_bit_binary_codec():
    spec = v1.decode([0] * 84)
    assert spec.schema_version == 1
    assert spec.attention_fraction == 0.0
    assert all(block.family == "cnn" for block in spec.flat_blocks)

    with pytest.raises(ValueError, match="84"):
        v1.decode([0] * 93)
    malformed = [0] * 84
    malformed[10] = 2
    with pytest.raises(ValueError, match="binary"):
        v1.decode(malformed)


def test_v1_and_v2_types_are_not_aliases():
    from bass import v2

    assert v1.ArchitectureSpec is not v2.ArchitectureSpec
    assert v1.BlockGene is not v2.BlockGene
