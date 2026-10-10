from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    api_key: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_dimensions: int = 384
    llm_provider: Literal["openai", "gemini"] = "openai"
    openai_api_key: str = ""
    gemini_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    jwt_secret: str = ""
    bootstrap_workspace_name: str = "Local legal workspace"
    bootstrap_admin_email: str = "admin@localhost.local"
    bootstrap_admin_password: str = ""
    retrieval_candidates: int = 30
    retrieval_results: int = 6
    max_upload_bytes: int = 25 * 1024 * 1024
    rag_eval_mode: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
