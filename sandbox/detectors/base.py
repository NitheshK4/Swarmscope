from abc import ABC, abstractmethod
from typing import List, Tuple
from sandbox.schemas import Message, ScenarioConfig

class BaseDetector(ABC):
    @abstractmethod
    def analyze(self, messages: List[Message], scenario: ScenarioConfig) -> Tuple[float, str]:
        """Analyzes conversation messages and returns a Tuple of (score 0.0-1.0, explanation_string)."""
        pass
