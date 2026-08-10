"""
Centralized app configuration, loaded from environment / .env via pydantic-settings.
Everything else in the app imports `settings` from here rather than reading
os.environ directly.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"

    # Database
    database_url: str = "sqlite+aiosqlite:///./smartreco.db"

    # Auth
    secret_key: str = "dev-only-insecure-key"
    access_token_expire_minutes: int = 60 * 24

    # LLM gateway (OpenAI-compatible) — currently OpenRouter
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_chat_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_embedding_model: str = "openai/text-embedding-3-small"

    # Vector store
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "products"

    # LangSmith (observability)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "smartreco-agent"

    mock_embeddings: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()