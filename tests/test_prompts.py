from pathlib import Path

import pytest

from chatbot.llm import build_messages
from chatbot.prompts import list_profiles, load_profile


def test_default_profile_builds_multishot_prefix():
    profile = load_profile("default")
    msgs = profile.to_messages()
    assert msgs[0]["role"] == "system"
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * len(profile.examples)


def test_build_messages_appends_history_and_user_turn():
    profile = load_profile("default")
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    msgs = build_messages(profile, history, "how are you?")
    assert msgs[-1] == {"role": "user", "content": "how are you?"}
    assert msgs[-3:-1] == history


def test_list_profiles_contains_default():
    assert "default" in list_profiles()


def test_missing_system_key_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("examples: []\n")
    with pytest.raises(ValueError):
        load_profile(bad)
