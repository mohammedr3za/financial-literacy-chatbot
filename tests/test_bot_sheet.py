"""The hand-testing core set is balanced over personas and framings (bench/make_bot_sheet.py, SPEC 8)."""
import collections
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.make_bot_sheet import core_set  # noqa: E402


def test_core_set_covers_every_topic_and_framing_with_balanced_personas():
    data = yaml.safe_load((ROOT / "bench/questions.yaml").read_text(encoding="utf-8"))
    core = core_set(data)
    assert len(core) == 3 * len(data["questions"])
    assert set(collections.Counter(c["persona"] for c in core).values()) == {len(data["questions"])}
    assert all(len({c["framing"] for c in core if c["question_id"] == q["id"]}) == 3 for q in data["questions"])
    assert all(c["question"] == data["questions"][[q["id"] for q in data["questions"]].index(c["question_id"])]["variants"][c["persona"]][c["framing"]]
               for c in core)
