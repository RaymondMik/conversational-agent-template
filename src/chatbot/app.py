"""Gradio chat UI with streaming responses, selectable prompt profiles and usage totals."""

from collections.abc import AsyncIterator
from typing import Any

import gradio as gr

from chatbot.config import settings
from chatbot.llm import build_messages, stream_completion
from chatbot.prompts import Message, list_profiles, load_profile

DESCRIPTION = "Model-agnostic chatbot on LiteLLM. Pick a prompt profile and model below."
EMPTY_TOTALS = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def format_stats(totals: dict) -> str:
    return (
        f"**Input tokens:** {totals['input_tokens']:,} &nbsp;·&nbsp; "
        f"**Output tokens:** {totals['output_tokens']:,} &nbsp;·&nbsp; "
        f"**Total cost:** ${totals['cost_usd']:.4f}"
    )


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
    totals: dict | None,
) -> AsyncIterator[tuple[str, Any, Any]]:
    """Yield (reply_so_far, stats_markdown, totals). Stats update only on the final yield."""
    totals = dict(totals or EMPTY_TOTALS)
    profile = load_profile(profile_name)
    messages = build_messages(profile, _clean_history(history), message)
    stream = stream_completion(messages, model=model.strip(), temperature=temperature)
    partial = ""
    try:
        async for delta in stream:
            partial += delta
            yield partial, gr.skip(), gr.skip()
    except Exception as exc:  # noqa: BLE001 - surface provider errors in the chat
        partial += f"\n\n**Error:** `{type(exc).__name__}: {exc}`"
    if stream.usage:
        totals["input_tokens"] += stream.usage.input_tokens
        totals["output_tokens"] += stream.usage.output_tokens
        totals["cost_usd"] += stream.usage.cost_usd
    yield partial, format_stats(totals), totals


def build_app() -> gr.Blocks:
    profiles = list_profiles()
    with gr.Blocks(title="Conversational Agent") as demo:
        gr.Markdown("# Conversational Agent")
        with gr.Row():
            gr.Markdown(DESCRIPTION)
            stats = gr.Markdown(format_stats(EMPTY_TOTALS))
        totals = gr.State(dict(EMPTY_TOTALS))  # per browser session
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[
                gr.Dropdown(choices=profiles, value=settings.prompt_profile, label="Prompt profile"),
                gr.Textbox(value=settings.llm_model, label="Model (LiteLLM format: provider/model)"),
                gr.Slider(0.0, 2.0, value=settings.llm_temperature, step=0.1, label="Temperature"),
                totals,
            ],
            additional_outputs=[stats, totals],
            additional_inputs_accordion=gr.Accordion("Settings", open=False),
        )
    return demo


def main() -> None:
    build_app().launch(server_name=settings.gradio_server_name, server_port=settings.gradio_server_port)


if __name__ == "__main__":
    main()
