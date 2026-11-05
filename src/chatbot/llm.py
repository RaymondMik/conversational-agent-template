"""Thin streaming wrapper over LiteLLM. Swap models by changing the model string."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import litellm

from chatbot.config import settings
from chatbot.prompts import Message, PromptProfile

litellm.drop_params = True  # silently drop params a given provider does not support


@dataclass
class Usage:
    """Token counts and cost for one completion. Cost is 0 for local or unpriced models."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def build_messages(profile: PromptProfile, history: list[Message], user_message: str) -> list[Message]:
    """System + few-shot examples + prior conversation + the new user turn."""
    return [*profile.to_messages(), *history, {"role": "user", "content": user_message}]


class CompletionStream:
    """Async iterable of text deltas. After iteration finishes, ``usage`` is populated."""

    def __init__(self, messages: list[Message], model: str, **litellm_kwargs) -> None:
        self.messages = messages
        self.model = model
        self.litellm_kwargs = litellm_kwargs
        self.usage: Usage | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        response = await litellm.acompletion(
            model=self.model, messages=self.messages, stream=True, **self.litellm_kwargs
        )
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        self.usage = _usage_from_chunks(chunks, self.messages)


def _usage_from_chunks(chunks: list, messages: list[Message]) -> Usage:
    """Rebuild the full response from stream chunks to read token usage and price it."""
    try:
        rebuilt = litellm.stream_chunk_builder(chunks, messages=messages)
    except Exception:  # noqa: BLE001 - usage is best-effort, never break the chat
        return Usage()
    if rebuilt is None or rebuilt.usage is None:
        return Usage()
    usage = Usage(
        input_tokens=rebuilt.usage.prompt_tokens or 0,
        output_tokens=rebuilt.usage.completion_tokens or 0,
    )
    try:
        usage.cost_usd = float(litellm.completion_cost(completion_response=rebuilt) or 0.0)
    except Exception:  # noqa: BLE001 - unknown pricing (e.g. Ollama) counts as free
        usage.cost_usd = 0.0
    return usage


def stream_completion(
    messages: list[Message],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **litellm_kwargs,
) -> CompletionStream:
    """Return a stream of assistant text deltas; read ``.usage`` once it is exhausted.

    Extra keyword arguments are passed straight to ``litellm.acompletion`` (for
    example provider-specific options, or ``mock_response`` in tests).
    """
    return CompletionStream(
        messages,
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        api_base=settings.llm_api_base,
        stream_options={"include_usage": True},
        **litellm_kwargs,
    )
