import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import yaml
from pydantic import ValidationError

from sandbox.schemas import ScenarioConfig


class ScenarioValidationError(ValueError):
    """Custom exception raised when scenario YAML parsing/validation fails."""
    pass

def validate_scenario_dict(data: Dict[str, Any]) -> ScenarioConfig:
    """Validates scenario raw dictionary and raises ScenarioValidationError with actionable errors."""
    if not isinstance(data, dict):
        raise ScenarioValidationError("Scenario YAML content must be a dictionary.")

    if not data.get("name"):
        raise ScenarioValidationError("Invalid scenario: 'name' is required and cannot be empty.")
    if not data.get("description"):
        raise ScenarioValidationError("Invalid scenario: 'description' is required and cannot be empty.")

    agents = data.get("agents")
    if not agents or not isinstance(agents, list):
        raise ScenarioValidationError("Invalid scenario: 'agents' must be a non-empty list of agent configurations.")
    if len(agents) < 2:
        raise ScenarioValidationError(f"Invalid scenario: At least 2 agents are required, found {len(agents)}.")

    for i, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ScenarioValidationError(f"Invalid agent at index {i}: Agent definition must be a dictionary.")
        if not agent.get("name"):
            raise ScenarioValidationError(f"Invalid agent at index {i}: Agent 'name' is required.")
        if not agent.get("role"):
            raise ScenarioValidationError(f"Invalid agent at index {i}: Agent 'role' is required.")
        if not agent.get("goal"):
            raise ScenarioValidationError(f"Invalid agent at index {i}: Agent 'goal' is required.")

    max_turns = data.get("max_turns", 15)
    if not isinstance(max_turns, int) or max_turns <= 0:
        raise ScenarioValidationError(f"Invalid scenario: 'max_turns' must be a positive integer, got {max_turns}.")

    try:
        return ScenarioConfig(**data)
    except ValidationError as e:
        raise ScenarioValidationError(f"Scenario validation failed: {str(e)}") from e

def load_scenario(file_path: str) -> ScenarioConfig:
    """Loads and validates a YAML scenario config file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise ScenarioValidationError(f"Failed to read or parse YAML file '{file_path}': {str(exc)}") from exc

    return validate_scenario_dict(data)

def generate_id() -> str:
    """Generates a unique simulation run ID."""
    return str(uuid.uuid4())[:8]

def get_current_timestamp() -> str:
    """Gets the current UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()

