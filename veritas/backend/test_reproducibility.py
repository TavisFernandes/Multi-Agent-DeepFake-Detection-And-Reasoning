"""
Byte-level reproducibility: every SHA-256 recorded in independent_evaluation_protocol.md matches the checked-out
file (line endings pinned in .gitattributes), and the manifest builders regenerate their committed bytes.
Run: python -m pytest test_reproducibility.py   (builder checks skip when the dataset is not on disk)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "training"))

import build_manifest_v2  # noqa: E402

_PROTOCOL = _HERE / "training" / "independent_evaluation_v1" / "independent_evaluation_protocol.md"
_RECORDED = re.findall(r"\| `([^`]+)` \| `([0-9A-F]{64})` \|", _PROTOCOL.read_text(encoding="utf-8"))
_DATASET = Path(json.loads(build_manifest_v2.V1_PATH.read_bytes())["metadata"]["dataset_root"])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.mark.parametrize("rel,recorded", _RECORDED, ids=[r for r, _ in _RECORDED])
def test_protocol_hash(rel, recorded):
    path = _HERE / rel
    if not path.exists():
        pytest.skip("untracked artifact not on this machine")
    assert sha(path) == recorded.lower()


@pytest.mark.skipif(not _DATASET.exists(), reason="FakeAVCeleb dataset not on disk")
def test_v2_builder_reproduces_committed_bytes(tmp_path):
    build_manifest_v2.main(tmp_path / "v2.json", tmp_path / "excl.json")
    assert sha(tmp_path / "v2.json") == sha(build_manifest_v2.V2_PATH)
    assert sha(tmp_path / "excl.json") == sha(build_manifest_v2.EXCLUSIONS_PATH)
