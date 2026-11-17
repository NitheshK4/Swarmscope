# ==============================================================================
# FILE: sandbox/detectors/base.py
# LOGIC: Abstract interface for all SwarmScope vulnerability detectors.
#        Requires each detector to implement analyze(messages, scenario), returning
#        a normalized risk score (0.0 to 1.0) and an interpretable natural language explanation.
# ==============================================================================

from abc import ABC, abstractmethod
from typing import List, Tuple

from sandbox.schemas import Message, ScenarioConfig


# ==============================================================================
# CORE DETECTOR CONTRACT
# ==============================================================================
class BaseDetector(ABC):
    @abstractmethod
    def analyze(self, messages: List[Message], scenario: ScenarioConfig) -> Tuple[float, str]:
        """Analyzes conversation messages and returns a Tuple of (score 0.0-1.0, explanation_string)."""
        pass
