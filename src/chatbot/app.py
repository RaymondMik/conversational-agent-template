"""Gradio chat UI with streaming responses and selectable prompt profiles."""

from collections.abc import AsyncIterator

import gradio as gr

from chatbot.config import settings
from chatbot.llm import build_messages, stream_completion
from chatbot.prompts import Message, list_profiles, load_profile


def _clean_history(history: list[dict]) -> list[Message]:
    """Keep only role/content text turns from Gradio's message history."""
    cleaned: list[Message] = []
    for msg in history:
        content = msg.get("content")
        if isinstance(content, str) and msg.get("role") in ("user", "assistant"):
            cleaned.append({"role": msg["role"], "content": content})
    return cleaned


async def respond(
    message: str,
    history: list[dict],
    profile_name: str,
    model: str,
    temperature: float,
) -> AsyncIterator[str]:
    profile = load_profile(profile_name)
    messages = build_messages(profile, _clean_history(history), message)
    partial = ""
    try:
        async for delta in stream_completion(messages, model=model.strip(), temperature=temperature):
            partial += delta
            yield partial
    except Exception as exc:  # noqa: BLE001 - surface provider errors in the chat
        yield f"{partial}\n\n**Error:** `{type(exc).__name__}: {exc}`"


def build_app() -> gr.Blocks:
    profiles = list_profiles()
    return gr.ChatInterface(
        fn=respond,
        title="Conversational Agent",
        description="Model-agnostic chatbot on LiteLLM. Pick a prompt profile and model below.",
        additional_inputs=[
            gr.Dropdown(choices=profiles, value=settings.prompt_profile, label="Prompt profile"),
            gr.Textbox(value=settings.llm_model, label="Model (LiteLLM format: provider/model)"),
            gr.Slider(0.0, 2.0, value=settings.llm_temperature, step=0.1, label="Temperature"),
        ],
        additional_inputs_accordion=gr.Accordion("Settings", open=False),
    )


def main() -> None:
    build_app().launch(server_name=settings.gradio_server_name, server_port=settings.gradio_server_port)


if __name__ == "__main__":
    main()
