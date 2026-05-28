from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    app_name: str = "FinSight"
    environment: str = "local"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    streamlit_api_base_url: str = "http://127.0.0.1:8000"

    groq_api_key: str = ""
    generation_model: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias=AliasChoices("GENERATION_MODEL", "GROQ_MODEL"),
    )
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chroma_dir: Path = Field(
        default=Path("data/chroma"),
        validation_alias=AliasChoices("CHROMA_DIR", "CHROMA_PERSIST_DIR"),
    )
    upload_dir: Path = Path("data/uploads")
    chunk_size: int = Field(default=1000, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k_retrieval: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias=AliasChoices("TOP_K_RETRIEVAL", "TOP_K"),
    )
    max_context_chars: int = Field(default=12000, ge=1000)
    generation_timeout_seconds: int = Field(default=30, ge=5)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chroma_persist_dir(self) -> Path:
        """Backward-compatible name for Chroma persistence path."""

        return self.chroma_dir

    @property
    def top_k(self) -> int:
        """Backward-compatible name for retrieval depth."""

        return self.top_k_retrieval

    @property
    def groq_model(self) -> str:
        """Backward-compatible name for the Groq generation model."""

        return self.generation_model


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so callers share one validated config object."""

    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
