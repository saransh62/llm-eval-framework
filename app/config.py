"""Configuration loaded from config.yaml with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "app" / "corpus"
CONFIG_PATH = ROOT / "config.yaml"


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    effort: str


@dataclass(frozen=True)
class Settings:
    llm: ModelConfig
    judge: ModelConfig
    top_k: int
    chunk_max_chars: int
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    def price(self, model: str) -> dict[str, float]:
        return self.pricing.get(model, {"input": 0.0, "output": 0.0})


def _model_config(section: dict, prefix: str, default_provider: str) -> ModelConfig:
    provider = os.getenv(f"{prefix}_PROVIDER") or section.get("provider") or default_provider
    model = os.getenv(f"{prefix}_MODEL") or section["models"][provider]
    return ModelConfig(
        provider=provider,
        model=model,
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", section["temperature"])),
        max_tokens=int(os.getenv(f"{prefix}_MAX_TOKENS", section["max_tokens"])),
        effort=os.getenv(f"{prefix}_EFFORT", section["effort"]),
    )


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    raw = yaml.safe_load(path.read_text())
    llm = _model_config(raw["llm"], "LLM", "fake")
    judge = _model_config(raw["judge"], "JUDGE", llm.provider)
    return Settings(
        llm=llm,
        judge=judge,
        top_k=int(os.getenv("TOP_K", raw["retrieval"]["top_k"])),
        chunk_max_chars=int(raw["retrieval"]["chunk_max_chars"]),
        pricing=raw.get("pricing", {}),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
