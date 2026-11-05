"""Runtime settings, loaded from environment variables and an optional .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_model: str = "anthropic/claude-sonnet-5-5"
    llm_api_base: str | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    prompt_profile: str = "default"

    gradio_server_name: str = "127.0.0.1"
    gradio_server_port: int = 7860


settings = Settings()
