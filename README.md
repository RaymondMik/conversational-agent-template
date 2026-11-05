# Conversational Agent Template

Model-agnostic chatbot scaffold: **LiteLLM** for the model layer, **Gradio** for the UI,
streaming responses, and YAML prompt profiles with multishot (few-shot) examples.

## Quick start

```bash
uv sync --extra dev
cp .env.example .env        # set LLM_MODEL and the matching API key
uv run chatbot              # opens http://127.0.0.1:7860
```

## Switching models

Set `LLM_MODEL` to any LiteLLM model string, or type one in the UI settings panel:

| Provider        | Example                          | Key / env               |
|-----------------|----------------------------------|-------------------------|
| Anthropic       | `anthropic/claude-sonnet-5-5`    | `ANTHROPIC_API_KEY`     |
| OpenAI          | `openai/gpt-4o`                  | `OPENAI_API_KEY`        |
| Google          | `gemini/gemini-2.5-pro`          | `GEMINI_API_KEY`        |
| Ollama (local)  | `ollama/llama3.1`                | `LLM_API_BASE`          |
| OpenAI-compatible | `openai/<model>`               | `LLM_API_BASE` + key    |

See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for more.

## Prompt profiles (multishot system prompts)

Each file in `prompts/` is a profile:

```yaml
name: Default assistant
system: |
  You are a concise, helpful assistant.
examples:
  - user: What is the capital of Australia?
    assistant: Canberra.
```

The system prompt is sent first, then each example as a user/assistant pair, then the
live conversation. Add a new `.yaml` file and it appears in the UI dropdown.

## Layout

```
src/chatbot/
  config.py   settings from .env / environment
  prompts.py  YAML profiles -> message prefix
  llm.py      LiteLLM streaming wrapper
  app.py      Gradio ChatInterface
prompts/      prompt profiles
tests/        pytest
```

## Development

```bash
uv run pytest
uv run ruff check .
```
