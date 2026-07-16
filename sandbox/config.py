import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Config:
    LLM_BACKEND = os.getenv("LLM_BACKEND", "dummy").lower()
    
    # Ollama Backend Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    
    # OpenAI Backend Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Anthropic Backend Settings
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
    
    # Storage Settings
    DUCKDB_PATH = os.getenv("DUCKDB_PATH", "simulation_runs.duckdb")
    
    @classmethod
    def validate_keys(cls):
        """Validates that key APIs are available depending on selected backend."""
        if cls.LLM_BACKEND == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set but 'openai' backend is selected.")
        if cls.LLM_BACKEND == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set but 'anthropic' backend is selected.")
