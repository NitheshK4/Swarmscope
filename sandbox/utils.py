import yaml
import uuid
from datetime import datetime, timezone
from sandbox.schemas import ScenarioConfig

def load_scenario(file_path: str) -> ScenarioConfig:
    """Loads and validates a YAML scenario config file."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return ScenarioConfig(**data)

def generate_id() -> str:
    """Generates a unique simulation run ID."""
    return str(uuid.uuid4())[:8]

def get_current_timestamp() -> str:
    """Gets the current UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()
