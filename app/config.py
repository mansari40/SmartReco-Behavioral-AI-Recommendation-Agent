"""
Centralized app configuration, loaded from environment / .env via pydantic-settings.
Everything else in the app imports `settings` from here rather than reading
os.environ directly.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"

    # Database
    database_url: str = "sqlite+aiosqlite:///./upulse.db"

    # Auth
    secret_key: str = "dev-only-insecure-key"
    access_token_expire_minutes: int = 60 * 24

    # Admin bootstrap — the single intended admin account, auto-created
    # idempotently at startup (never hardcoded in source).
    admin_email: str = Field(default="upulse@admin.com", env="ADMIN_EMAIL")
    admin_password: str = Field(default="", env="ADMIN_PASSWORD")

    # LLM gateway (OpenRouter local/dev). Keep the provider/model configurable.
    llm_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    llm_base_url: str = Field(default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL")
    llm_chat_model: str = Field(default="openrouter/free", validation_alias="OPENROUTER_MODEL")
    llm_embedding_model: str = Field(default="openai/text-embedding-3-small", validation_alias="OPENROUTER_EMBEDDING_MODEL")
    llm_max_tokens: int = Field(default=4096, validation_alias="OPENROUTER_MAX_TOKENS")

    # Email / password reset
    smtp_host: str = Field(default="", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: str = Field(default="", env="SMTP_USER")
    smtp_password: str = Field(default="", env="SMTP_PASSWORD")
    smtp_from: str = Field(default="no-reply@upulse.ai", env="SMTP_FROM")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")

    # Public base URL used to build absolute links in transactional emails
    # (e.g. password reset). On Render this is injected automatically from the
    # service URL; locally it defaults to the dev server.
    public_base_url: str = Field(default="http://localhost:8000", env="PUBLIC_BASE_URL")

    # Vector store
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "products"

    # LangSmith (observability)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "upulse-agent"

    mock_embeddings: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()