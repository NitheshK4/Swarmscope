from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

class Message(BaseModel):
    sender: str
    receiver: str
    content: str
    turn: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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

class SimulationMetadata(BaseModel):
    simulation_id: str
    scenario_name: str
    timestamp: str
    total_turns: int
    backend: str
    temperature: float
    status: str

class SimulationRun(BaseModel):
    metadata: SimulationMetadata
    messages: List[Message]
    detector_scores: Dict[str, float] = Field(default_factory=dict)
    detector_explanations: Dict[str, str] = Field(default_factory=dict)
