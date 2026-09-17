from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

API_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=API_DIR / ".env", extra="ignore")

    database_url: str = "postgresql://portfolio:portfolio@localhost:5433/portfolio"
    cors_origins: str = "http://localhost:8080"

    # Bearer token for PUT /admin/content; empty disables the endpoint.
    admin_token: str = ""

    # Embeddings: same model in both backends so indexed and query vectors match.
    embedding_backend: Literal["local", "http"] = "local"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    embedding_api_url: str = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
    embedding_api_token: str = ""

    text_search_config: str = "spanish"

    # Search
    results_limit: int = 5
    candidate_pool: int = 30
    rrf_k: int = 60

    # Confidence bands on the margin (best − mean similarity), calibrated with scripts/calibrate.py
    margin_high: float = 0.075
    margin_medium: float = 0.046
    gap_high: float = 0.045

    # Rocchio
    rocchio_alpha: float = 0.7
    rocchio_beta: float = 0.8

    event_retention_days: int = 180

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
