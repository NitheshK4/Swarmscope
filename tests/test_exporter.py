import pytest
import json
from sandbox.schemas import Message, SimulationRun, SimulationMetadata
from sandbox.analytics.exporter import ConversationExporter


@pytest.fixture
def sample_run():
    metadata = SimulationMetadata(
        simulation_id="export_test_001",
        scenario_name="negotiation",
        timestamp="2026-07-15T00:00:00",
        total_turns=3,
        backend="dummy",
        temperature=0.7,
        status="completed"
    )
    messages = [
        Message(sender="Alice", receiver="Bob", content="I'd like to buy the car for $10,000.", turn=1),
        Message(sender="Bob", receiver="Alice", content="That's too low. How about $13,000?", turn=2),
        Message(sender="Alice", receiver="Bob", content="Let's settle on $11,500. Deal.", turn=3),
    ]
    return SimulationRun(
        metadata=metadata,
        messages=messages,
        detector_scores={"loop": 0.1, "deadlock": 0.2, "collusion": 0.0},
        detector_explanations={"loop": "No loop", "deadlock": "Minor", "collusion": "None"}
    )


def test_export_json(sample_run):
    """Verify JSON export produces valid JSON with expected structure."""
    exporter = ConversationExporter()
    result = exporter.to_json(sample_run)

    data = json.loads(result)
    assert data["metadata"]["simulation_id"] == "export_test_001"
    assert len(data["messages"]) == 3
    assert data["messages"][0]["sender"] == "Alice"
    assert "sentiment_score" in data["messages"][0]


def test_export_csv(sample_run):
    """Verify CSV export produces valid CSV with header and data rows."""
    exporter = ConversationExporter()
    result = exporter.to_csv(sample_run)

    lines = result.strip().split("\n")
    assert len(lines) == 4  # 1 header + 3 data rows
    assert "simulation_id" in lines[0]
    assert "Alice" in lines[1]


def test_export_jsonl(sample_run):
    """Verify JSONL export produces one JSON object per line."""
    exporter = ConversationExporter()
    result = exporter.to_jsonl(sample_run)

    lines = [line for line in result.strip().split("\n") if line]
    assert len(lines) == 4  # 1 metadata + 3 messages

    # First line should be metadata
    meta = json.loads(lines[0])
    assert meta["type"] == "metadata"
    assert meta["simulation_id"] == "export_test_001"

    # Remaining lines should be messages
    msg1 = json.loads(lines[1])
    assert msg1["type"] == "message"
    assert msg1["sender"] == "Alice"


def test_export_yaml(sample_run):
    """Verify YAML export produces string containing key fields."""
    exporter = ConversationExporter()
    result = exporter.export(sample_run, fmt="yaml")
    assert "simulation_id: export_test_001" in result or "simulation_id" in result
    assert "Alice" in result

def test_export_markdown(sample_run):
    """Verify Markdown export produces structured markdown document."""
    exporter = ConversationExporter()
    result = exporter.export(sample_run, fmt="markdown")
    assert "# Simulation Run: export_test_001" in result
    assert "## Detector Scores" in result
    assert "## Message Transcript" in result
    assert "Alice -> Bob" in result

def test_export_invalid_format(sample_run):
    """Verify that unsupported format raises ValueError."""
    exporter = ConversationExporter()

    with pytest.raises(ValueError, match="Unsupported"):
        exporter.export(sample_run, fmt="xml")

