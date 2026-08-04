from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
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

    @abstractmethod
    def generate(self, system_prompt: str, prompt: str, history: Optional[List] = None, temperature: float = 0.7) -> str:
        """Generates a response text from the LLM given system prompt, current prompt, and history."""
        pass

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
