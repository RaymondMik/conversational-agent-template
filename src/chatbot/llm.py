"""Thin streaming wrapper over LiteLLM. Swap models by changing the model string."""

from collections.abc import AsyncIterator

import litellm

from chatbot.config import settings
from chatbot.prompts import Message, PromptProfile

litellm.drop_params = True  # silently drop params a given provider does not support


def build_messages(profile: PromptProfile, history: list[Message], user_message: str) -> list[Message]:
    """System + few-shot examples + prior conversation + the new user turn."""
    return [*profile.to_messages(), *history, {"role": "user", "content": user_message}]


async def stream_completion(
    messages: list[Message],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **litellm_kwargs,
) -> AsyncIterator[str]:
    """Yield the assistant reply as text deltas.

    Extra keyword arguments are passed straight to ``litellm.acompletion`` (for
    example provider-specific options, or ``mock_response`` in tests).
    """
    response = await litellm.acompletion(
        model=model or settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        api_base=settings.llm_api_base,
        stream=True,
        **litellm_kwargs,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
