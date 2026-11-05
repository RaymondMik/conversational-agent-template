"""Prompt profiles: a system prompt plus multishot (few-shot) example turns.

A profile is a YAML file in the prompts/ directory:

    name: Friendly helper
    system: You are ...
    examples:
      - user: ...
        assistant: ...

`to_messages()` turns a profile into the prefix of an OpenAI-style message list:
the system message followed by alternating user/assistant example turns.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from chatbot.config import PROMPTS_DIR

Message = dict[str, str]


@dataclass
class Example:
    user: str
    assistant: str


@dataclass
class PromptProfile:
    name: str
    system: str
    examples: list[Example] = field(default_factory=list)

    def to_messages(self) -> list[Message]:
        messages: list[Message] = [{"role": "system", "content": self.system.strip()}]
        for ex in self.examples:
            messages.append({"role": "user", "content": ex.user.strip()})
            messages.append({"role": "assistant", "content": ex.assistant.strip()})
        return messages


def load_profile(name_or_path: str | Path, prompts_dir: Path = PROMPTS_DIR) -> PromptProfile:
    """Load a profile by name (looked up in prompts_dir) or by explicit path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = prompts_dir / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "system" not in data:
        raise ValueError(f"Prompt profile {path} has no 'system' key")
    examples = [Example(user=e["user"], assistant=e["assistant"]) for e in data.get("examples", [])]
    return PromptProfile(name=data.get("name", path.stem), system=data["system"], examples=examples)


def list_profiles(prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    return sorted(p.stem for p in prompts_dir.glob("*.yaml"))
