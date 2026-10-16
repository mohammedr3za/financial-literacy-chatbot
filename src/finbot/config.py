"""Settings from environment variables (.env supported). See SPEC section 2."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.getenv("FINBOT_MODEL", "qwen3:8b"))
    backend: str = field(default_factory=lambda: os.getenv("FINBOT_BACKEND", "ollama"))
    base_url: str = field(default_factory=lambda: os.getenv("FINBOT_BASE_URL", "http://localhost:11434"))
    thinking: bool = field(default_factory=lambda: os.getenv("FINBOT_THINKING", "off").lower() == "on")
    num_ctx: int = field(default_factory=lambda: int(os.getenv("FINBOT_NUM_CTX", "16384")))
    temperature: float = field(default_factory=lambda: float(os.getenv("FINBOT_TEMPERATURE", "0.0")))
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv("FINBOT_MAX_OUTPUT_TOKENS", "700")))
    embed_model: str = field(default_factory=lambda: os.getenv("FINBOT_EMBED_MODEL", "bge-m3"))
    embed_url: str = field(default_factory=lambda: os.getenv("FINBOT_EMBED_URL", "http://localhost:11434"))
    kb_dir: Path = PROJECT_ROOT / "kb"
    top_k: int = 3
    min_dense_score: float = field(default_factory=lambda: float(os.getenv("FINBOT_MIN_DENSE_SCORE", "0.50")))
    max_tool_iterations: int = 6
    max_revisions: int = 1
    session_cap: int = 25


def get_settings() -> Settings:
    return Settings()
