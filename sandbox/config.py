# ==============================================================================
# FILE: sandbox/config.py
# LOGIC: Centralized configuration manager for SwarmScope.
#        Loads environment variables from .env, defines defaults for LLM backends,
#        API keys, models, DuckDB paths, and webhook thresholds, and provides
#        type-safe getters and validation helpers.
# ==============================================================================

import os

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ==============================================================================
# CORE CONFIGURATION SPECIFICATION
# ==============================================================================
class Config:
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "dummy").lower()

    # Ollama Backend Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    # OpenAI Backend Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Anthropic Backend Settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    # Storage Settings
    DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", "simulation_runs.duckdb")

    # Webhook / Notification Settings
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    WEBHOOK_ENABLED: bool = os.getenv("WEBHOOK_ENABLED", "false").lower() in ("true", "1", "yes")

    @classmethod
    def get_float(cls, key: str, default: float) -> float:
        val = os.getenv(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    @classmethod
    def get_int(cls, key: str, default: int) -> int:
        val = os.getenv(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    @classmethod
    def risk_alert_threshold(cls) -> float:
        return cls.get_float("RISK_ALERT_THRESHOLD", 0.6)

    # ==========================================================================
    # CORE LOGIC: API Key / Backend validation check
    # ==========================================================================
    @classmethod
    def validate_keys(cls):
        """Validates that key APIs are available depending on selected backend."""
        if cls.LLM_BACKEND == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set but 'openai' backend is selected.")
        if cls.LLM_BACKEND == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set but 'anthropic' backend is selected.")
