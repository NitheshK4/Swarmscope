from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    sender: str
    receiver: str
    content: str
    turn: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ConversationEvent(BaseModel):
    event_type: str = "message"
    turn: int
    sender: str
    receiver: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentConfig(BaseModel):
    name: str
    role: str
    goal: str
    traits: Optional[Dict[str, float]] = Field(default_factory=lambda: {"assertiveness": 0.5, "cooperativeness": 0.5})

class ScenarioConfig(BaseModel):
    name: str
    description: str
    agents: List[AgentConfig]
    system_prompt: str
    max_turns: int = 15

class SimulationConfig(BaseModel):
    scenario_name: str
    backend: str = "dummy"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_turns: int = Field(10, ge=1)
    random_seed: Optional[int] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class DetectorFinding(BaseModel):
    detector_name: str
    score: float = Field(0.0, ge=0.0, le=1.0)
    detected: bool = False
    severity: str = "LOW"
    explanation: str = ""
    evidence: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)

class SimulationMetadata(BaseModel):
    simulation_id: str
    scenario_name: str
    timestamp: str
    total_turns: int = Field(..., ge=0)
    backend: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    status: str = "COMPLETED"
    duration_ms: Optional[float] = 0.0
    random_seed: Optional[int] = None
    loop_score: Optional[float] = 0.0
    deadlock_score: Optional[float] = 0.0
    collusion_score: Optional[float] = 0.0
    goal_drift_score: Optional[float] = 0.0
    jailbreak_score: Optional[float] = 0.0
    escalation_score: Optional[float] = 0.0
    information_leakage_score: Optional[float] = 0.0

class SimulationRun(BaseModel):
    metadata: SimulationMetadata
    messages: List[Message]
    events: List[ConversationEvent] = Field(default_factory=list)
    detector_scores: Dict[str, float] = Field(default_factory=dict)
    detector_explanations: Dict[str, str] = Field(default_factory=dict)
    findings: List[DetectorFinding] = Field(default_factory=list)

    def get_peak_risk(self) -> float:
        """Returns the highest risk score among all detector evaluations."""
        if not self.detector_scores:
            return 0.0
        return max(self.detector_scores.values())

    def get_highest_severity(self) -> str:
        """Returns the highest severity label across all findings."""
        severities = {f.severity.upper() for f in self.findings}
        if "HIGH" in severities:
            return "HIGH"
        if "MEDIUM" in severities:
            return "MEDIUM"
        return "LOW"


