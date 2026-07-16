from abc import ABC, abstractmethod

class BaseLLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, prompt: str, history: list = None, temperature: float = 0.7) -> str:
        """Generates a response from the LLM given system prompt, current prompt, and history."""
        pass
