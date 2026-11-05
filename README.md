# Conversational Agent Template

A minimal, production-shaped starting point for a chatbot that is **not tied to any single
LLM provider**. Swap between Anthropic, OpenAI, Google, Mistral, a local Ollama model, or any
OpenAI-compatible server by changing one string. Responses stream token by token into a
Gradio web UI, and the assistant's behaviour is defined by YAML prompt profiles that combine a
system prompt with multishot (few-shot) examples.

## Contents

- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Switching models](#switching-models)
- [Prompt profiles and multishot prompting](#prompt-profiles-and-multishot-prompting)
- [Streaming](#streaming)
- [Token and cost tracking](#token-and-cost-tracking)
- [Project layout](#project-layout)
- [Development workflow](#development-workflow)
- [Extending the template](#extending-the-template)
- [Troubleshooting](#troubleshooting)

## Tech stack

| Layer | Technology | Why it was chosen |
|---|---|---|
| Language | **Python 3.11+** | Modern typing (`X \| None`, `dataclass`), async generators, wide ML ecosystem. |
| Package manager | **uv** | Fast, reproducible installs with a lockfile (`uv.lock`), no separate venv management. |
| Build backend | **hatchling** | Lightweight PEP 517 backend; `src/` layout keeps tests from importing uninstalled code. |
| LLM abstraction | **LiteLLM** | One `completion()` API for 100+ providers using OpenAI-style messages. Handles auth, retries, streaming, and parameter translation per provider. |
| Web UI | **Gradio 6** | `ChatInterface` gives a full chat UI (history, stop button, streaming rendering, extra controls) from a single Python function. |
| Configuration | **pydantic-settings** | Typed settings loaded from environment variables and `.env`, validated at startup. |
| Prompt storage | **PyYAML** | Prompt profiles live as human-editable YAML files, outside the code. |
| Testing | **pytest + pytest-asyncio** | Async tests for the streaming path; LiteLLM's `mock_response` lets tests run with no API key. |
| Linting | **ruff** | Single fast tool for linting and import ordering. |

### How the pieces fit together

- **LiteLLM** is the only thing that talks to model providers. The rest of the code never
  imports a provider SDK. Because LiteLLM normalises every provider to the OpenAI
  `messages=[{"role": ..., "content": ...}]` format, the prompt-building code is provider-neutral.
- **Gradio** owns the UI and the event loop. Its `ChatInterface` calls our `respond` function
  with the new message, the chat history (already in OpenAI message format), and the values of
  any extra controls. Because `respond` is an **async generator**, every `yield` repaints the
  assistant bubble, which is how streaming reaches the browser.
- **pydantic-settings** reads `.env` once at import time and exposes a `settings` object.
  LiteLLM reads provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) directly from the
  environment, so the template never has to handle secrets itself.
- **Prompt profiles** are plain data. `prompts.py` turns a YAML file into a list of messages
  that is prepended to every request.

## Architecture

```
┌──────────────┐   message, history,    ┌──────────────────┐
│  Browser     │   profile, model, temp │  app.py          │
│  (Gradio UI) │ ─────────────────────▶ │  respond()       │
│              │ ◀───────────────────── │  async generator │
└──────────────┘   partial text (yield) └────────┬─────────┘
                                                 │ build_messages()
                                                 ▼
                                    ┌──────────────────────────┐
                                    │ prompts.py               │
                                    │ YAML ─▶ PromptProfile    │
                                    │ system + few-shot turns  │
                                    └────────────┬─────────────┘
                                                 │ [system, ex_u, ex_a, ..., history..., user]
                                                 ▼
                                    ┌──────────────────────────┐      ┌─────────────────┐
                                    │ llm.py                   │      │ Provider API    │
                                    │ litellm.acompletion(     │ ───▶ │ Anthropic /     │
                                    │   stream=True)           │ ◀─── │ OpenAI / Ollama │
                                    │ yields text deltas       │      │ / ...           │
                                    └──────────────────────────┘      └─────────────────┘
```

### Request lifecycle

1. The user submits a message. Gradio calls `respond(message, history, profile, model, temperature)`.
2. `load_profile(profile)` reads `prompts/<profile>.yaml` and returns a `PromptProfile`.
3. `build_messages()` assembles the full message list:
   `[system] + [few-shot pairs] + [prior conversation] + [new user message]`.
4. `stream_completion()` calls `litellm.acompletion(..., stream=True)` and yields each
   `delta.content` chunk as it arrives.
5. `respond()` accumulates the chunks and yields the growing string. Gradio re-renders the
   assistant message on each yield.
6. Once the stream ends, token usage and cost are computed and added to the session totals
   shown in the header.
7. If the provider raises, the error is rendered inside the chat bubble rather than crashing
   the request.

## Quick start

```bash
# 1. Install dependencies (creates .venv and installs from uv.lock)
uv sync --extra dev

# 2. Configure
cp .env.example .env
#    Edit .env: set LLM_MODEL and the matching provider API key

# 3. Run
uv run chatbot
#    Open http://127.0.0.1:7860
```

If you do not have `uv`, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`
or use `pip install -e ".[dev]"` inside a virtualenv and run `python -m chatbot.app`.

## Configuration

All settings are read from environment variables, with `.env` as a convenient source.
Defaults live in `src/chatbot/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `anthropic/claude-sonnet-5-5` | Model in LiteLLM `provider/model` format. |
| `LLM_API_BASE` | unset | Base URL for self-hosted or OpenAI-compatible endpoints. |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature (overridable per message in the UI). |
| `LLM_MAX_TOKENS` | `2048` | Maximum tokens in the reply. |
| `PROMPT_PROFILE` | `default` | Profile selected on startup; file name without `.yaml`. |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Bind address. Use `0.0.0.0` inside containers. |
| `GRADIO_SERVER_PORT` | `7860` | Port. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, ... | unset | Provider credentials, read by LiteLLM. |

`.env` is gitignored. Commit `.env.example` instead.

## Switching models

Set `LLM_MODEL`, or type a model string in the **Settings** accordion in the UI. The string
format is always `provider/model`.

| Provider | Example `LLM_MODEL` | Required env |
|---|---|---|
| Anthropic | `anthropic/claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-pro` | `GEMINI_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| Groq | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Azure OpenAI | `azure/<deployment-name>` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| AWS Bedrock | `bedrock/anthropic.claude-...` | AWS credentials in env |
| Ollama (local) | `ollama/llama3.1` | `LLM_API_BASE=http://localhost:11434` |
| Any OpenAI-compatible server (vLLM, LM Studio, llama.cpp) | `openai/<model-name>` | `LLM_API_BASE=http://host:port/v1`, `OPENAI_API_KEY=anything` |

The full list is in the [LiteLLM provider docs](https://docs.litellm.ai/docs/providers).

`llm.py` sets `litellm.drop_params = True`, so parameters a provider does not support (for
example `temperature` on some reasoning models) are dropped silently instead of raising.

## Prompt profiles and multishot prompting

A **prompt profile** is a YAML file in `prompts/`. It pairs a system prompt with a list of
example exchanges. This is multishot (few-shot) prompting: the examples teach the model the
expected tone, format, and length far more reliably than instructions alone.

```yaml
# prompts/support.yaml
name: Support agent
system: |
  You are a support agent for Acme Widgets. Be warm, brief, and always end
  with a next step the customer can take.
examples:
  - user: My widget won't turn on.
    assistant: |
      Sorry to hear that. Please hold the power button for 10 seconds to reset it.
      If it still won't start, reply with your order number and I'll arrange a replacement.
  - user: Do you ship to Canada?
    assistant: Yes, shipping to Canada takes 5 to 7 business days. Want me to check the cost for your postcode?
```

### How it is sent to the model

`PromptProfile.to_messages()` produces:

```
system:    <system prompt>
user:      My widget won't turn on.
assistant: Sorry to hear that. ...
user:      Do you ship to Canada?
assistant: Yes, shipping to Canada ...
--- live conversation follows ---
user:      <history turn 1>
assistant: <history turn 1 reply>
user:      <new message>
```

The model sees the examples as prior turns of the same conversation, which is the standard
way to few-shot a chat model across all providers.

### Adding a profile

1. Create `prompts/<name>.yaml` with `system` (required), `name` and `examples` (optional).
2. Restart the app. The new profile appears in the **Prompt profile** dropdown.
3. Optionally set `PROMPT_PROFILE=<name>` in `.env` to make it the default.

Tips:

- Keep examples short and representative. Three to five is usually enough.
- Examples count against the context window on every request. Trim them for long chats.
- Use `|` block scalars in YAML for multi-line text to preserve newlines.

## Streaming

Streaming works end to end without any buffering layer:

- `llm.py` requests `stream=True` and iterates the async response, yielding only non-empty
  `delta.content` strings.
- `app.py`'s `respond` is an `async def` generator. Gradio treats each `yield` as a UI update.
- The **Stop** button in the UI cancels the generator, which closes the HTTP stream to the provider.

`stream_completion()` accepts arbitrary keyword arguments and forwards them to
`litellm.acompletion`. This is how tests inject `mock_response="..."` to exercise the
streaming path with no network and no key. It is also the place to pass provider-specific
options.

## Token and cost tracking

The header shows running totals for the current browser session: input tokens, output tokens,
and total cost in USD.

- After each streamed reply, `llm.py` rebuilds the response with `litellm.stream_chunk_builder`
  to read exact token usage. Providers that support it (OpenAI, Anthropic, ...) report usage
  in the stream; otherwise LiteLLM counts tokens locally.
- Cost comes from `litellm.completion_cost`, which uses LiteLLM's built-in price table. Local
  or unpriced models (for example Ollama) show `$0.0000`.
- Totals live in a `gr.State`, so each browser tab has its own counter. They reset on reload.

## Project layout

```
.
├── pyproject.toml          project metadata, dependencies, tool config, `chatbot` CLI entry point
├── uv.lock                 locked dependency graph
├── .env.example            documented template for .env
├── README.md
├── prompts/                prompt profiles (one YAML per profile)
│   ├── default.yaml
│   └── pirate.yaml
├── src/chatbot/
│   ├── __init__.py
│   ├── config.py           Settings model; PROJECT_ROOT and PROMPTS_DIR constants
│   ├── prompts.py          PromptProfile, load_profile(), list_profiles()
│   ├── llm.py              build_messages(), stream_completion()
│   └── app.py              respond(), build_app(), main()
└── tests/
    ├── test_prompts.py     profile loading and message assembly
    └── test_llm.py         streaming with LiteLLM mock provider
```

## Development workflow

```bash
uv run pytest              # run the test suite (no API key needed)
uv run ruff check .        # lint
uv run ruff format .       # format
uv run chatbot             # start the UI
```

Tests never hit a real provider. `tests/test_llm.py` passes `mock_response` through
`stream_completion`, so LiteLLM fabricates a streamed reply locally.

## Extending the template

- **Tool calling / function calling.** LiteLLM supports OpenAI-style `tools=[...]` across
  providers. Add a `tools` argument to `stream_completion`, inspect `delta.tool_calls` in the
  stream, and loop until the model stops requesting tools.
- **Per-user conversation persistence.** Gradio's `ChatInterface(save_history=True)` keeps
  history in the browser. For server-side storage, wrap `respond` and write `history` to a
  database keyed by session.
- **Multimodal input.** Set `ChatInterface(multimodal=True)`; `message` becomes a dict with
  `text` and `files`. Convert images to the `image_url` content blocks LiteLLM expects.
- **Model allowlist.** Replace the free-text model `Textbox` in `build_app()` with a
  `gr.Dropdown` of approved model strings.
- **Observability.** LiteLLM has built-in callbacks for Langfuse, OpenTelemetry, Helicone and
  others: set `litellm.success_callback = ["langfuse"]` in `llm.py`.
- **Retries and fallbacks.** Pass `num_retries=` and `fallbacks=[...]` through
  `stream_completion`'s kwargs, or use a LiteLLM `Router` for load balancing.
- **Deployment.** Set `GRADIO_SERVER_NAME=0.0.0.0` and run `uv run chatbot` in a container,
  or mount `build_app()` into FastAPI with `gr.mount_gradio_app`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AuthenticationError` in the chat bubble | Missing or wrong API key for the chosen provider | Set the matching `*_API_KEY` in `.env` and restart. |
| `BadRequestError: LLM Provider NOT provided` | `LLM_MODEL` lacks the `provider/` prefix | Use `anthropic/...`, `openai/...`, etc. |
| Connection refused with Ollama | Server not running or wrong base URL | Start `ollama serve`; set `LLM_API_BASE=http://localhost:11434`. |
| `TypeError: ChatInterface.__init__() got an unexpected keyword argument 'type'` | Code written for Gradio 5 running on Gradio 6 | Gradio 6 uses message-format history by default; remove `type="messages"`. |
| `Cannot find a function with api_name: /chat` when using `gradio_client` | Gradio 6 names the endpoint after the function | Use `api_name="/respond"`. |
| Profile missing from dropdown | File not in `prompts/` or wrong extension | Use `.yaml` and restart the app. |
| Reply cut off | `LLM_MAX_TOKENS` too low | Raise it in `.env` or per request. |
