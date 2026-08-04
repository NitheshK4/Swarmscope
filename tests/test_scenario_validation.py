import pytest
import tempfile
import os
from sandbox.utils import load_scenario, validate_scenario_dict, ScenarioValidationError

def test_valid_scenario_validation():
    valid_data = {
        "name": "valid_test",
        "description": "A valid test scenario",
        "agents": [
            {"name": "Alice", "role": "Buyer", "goal": "Buy stuff"},
            {"name": "Bob", "role": "Seller", "goal": "Sell stuff"},
        ],
        "system_prompt": "Simulate trade",
        "max_turns": 10,
    }
    config = validate_scenario_dict(valid_data)
    assert config.name == "valid_test"
    assert len(config.agents) == 2

def test_invalid_scenario_missing_fields():
    with pytest.raises(ScenarioValidationError, match="name"):
        validate_scenario_dict({"description": "No name scenario"})

    with pytest.raises(ScenarioValidationError, match="agents"):
        validate_scenario_dict({"name": "test", "description": "desc", "agents": []})

    with pytest.raises(ScenarioValidationError, match="At least 2 agents"):
        validate_scenario_dict({
            "name": "test",
            "description": "desc",
            "agents": [{"name": "Solo", "role": "Actor", "goal": "Do work"}]
        })

def test_invalid_agent_configuration():
    with pytest.raises(ScenarioValidationError, match="goal"):
        validate_scenario_dict({
            "name": "test",
            "description": "desc",
            "agents": [
                {"name": "Alice", "role": "Buyer", "goal": "Buy"},
                {"name": "Bob", "role": "Seller"},
            ]
        })

def test_invalid_turns_configuration():
    with pytest.raises(ScenarioValidationError, match="max_turns"):
        validate_scenario_dict({
            "name": "test",
            "description": "desc",
            "agents": [
                {"name": "Alice", "role": "Buyer", "goal": "Buy"},
                {"name": "Bob", "role": "Seller", "goal": "Sell"},
            ],
            "max_turns": -5
        })

def test_file_scenario_loading():
    yaml_content = """
name: temp_scenario
description: Temp scenario description
system_prompt: System prompt
max_turns: 5
agents:
  - name: Agent1
    role: Role1
    goal: Goal1
  - name: Agent2
    role: Role2
    goal: Goal2
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tf.write(yaml_content)
        tf_path = tf.name

    try:
        config = load_scenario(tf_path)
        assert config.name == "temp_scenario"
        assert config.max_turns == 5
    finally:
        os.remove(tf_path)
