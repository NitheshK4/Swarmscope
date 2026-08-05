from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    content: str
    backend_name: str
    model: str = "default"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Optional[Any] = None

class BackendError(Exception):
    """Exception raised when an LLM backend fails."""
    def __init__(self, backend_name: str, message: str, original_exception: Optional[Exception] = None):
        self.backend_name = backend_name
        self.message = message
        self.original_exception = original_exception
        super().__init__(f"[{backend_name}] {message}")

class BaseLLMBackend(ABC):
    backend_name: str = "base"
    model_name: str = "default"

    def __init__(self, max_retries: int = 3, backoff_factor: float = 0.5, timeout: float = 30.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    @abstractmethod
    def generate(self, system_prompt: str, prompt: str, history: Optional[List] = None, temperature: float = 0.7) -> str:
        """Generates a response text from the LLM given system prompt, current prompt, and history."""
        pass

    def with_retry(self, func, *args, **kwargs):
        """Executes func with exponential backoff on network/transient failures."""
        import time
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor * (2 ** attempt))
        raise BackendError(self.backend_name, f"Failed after {self.max_retries} attempts: {last_exc}", last_exc)

    def generate_response(
        self, system_prompt: str, prompt: str, history: Optional[List] = None, temperature: float = 0.7
    ) -> LLMResponse:
        """Normalized generator method that returns structured LLMResponse."""
        content = self.generate(system_prompt, prompt, history, temperature)
        return LLMResponse(
            content=content,
            backend_name=self.backend_name,
            model=self.model_name,
            metadata={"temperature": temperature},
        )

