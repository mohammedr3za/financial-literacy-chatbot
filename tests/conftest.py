import pytest


@pytest.fixture(autouse=True)
def _chatlog_in_tmp(tmp_path, monkeypatch):
    """Tests must never write to the real conversation log (SPEC 7.4)."""
    monkeypatch.setenv("FINBOT_CHATLOG_DIR", str(tmp_path / "chatlog"))
