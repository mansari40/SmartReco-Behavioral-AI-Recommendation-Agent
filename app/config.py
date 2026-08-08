"""
Centralized app configuration, loaded from environment / .env via pydantic-settings.
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

    # Mesh API — mandatory gateway for all LLM/embedding calls
    mesh_api_key: str = ""
    mesh_base_url: str = "https://api.meshapi.ai/v1"
    mesh_chat_model: str = "openai/gpt-4o"
    mesh_embedding_model: str = "openai/text-embedding-3-small"

    # Vector store
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "products"

    # LangSmith (observability)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "smartreco-agent"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()