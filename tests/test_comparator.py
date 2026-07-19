import os
from sandbox.schemas import SimulationRun, SimulationMetadata, Message
from sandbox.analytics.comparator import SimulationComparator
from sandbox.storage import get_store


def test_simulation_comparator():
    """Verify the simulation comparator produces correct comparison structure."""
    test_db = "test_comparator.duckdb"
    if os.path.exists(test_db):
        os.remove(test_db)

    try:
        store = get_store(test_db)

        # Create two runs with different risk profiles
        meta_a = SimulationMetadata(
            simulation_id="comp_a", scenario_name="negotiation",
            timestamp="2026-07-15T00:00:00", total_turns=4,
            backend="dummy", temperature=0.5, status="completed"
        )
        run_a = SimulationRun(
            metadata=meta_a,
            messages=[
                Message(sender="Alice", receiver="Bob", content="I offer $10,000.", turn=1),
                Message(sender="Bob", receiver="Alice", content="Counter: $13,000.", turn=2),
                Message(sender="Alice", receiver="Bob", content="How about $11,500?", turn=3),
                Message(sender="Bob", receiver="Alice", content="Deal at $11,500.", turn=4),
            ],
            detector_scores={"loop": 0.1, "deadlock": 0.2, "collusion": 0.0, "goal_drift": 0.05},
            detector_explanations={"loop": "Low", "deadlock": "Low", "collusion": "None", "goal_drift": "Low"}
        )

        meta_b = SimulationMetadata(
            simulation_id="comp_b", scenario_name="negotiation",
            timestamp="2026-07-15T00:01:00", total_turns=4,
            backend="dummy", temperature=1.0, status="completed"
        )
        run_b = SimulationRun(
            metadata=meta_b,
            messages=[
                Message(sender="Alice", receiver="Bob", content="Give me the car for $8,000!", turn=1),
                Message(sender="Bob", receiver="Alice", content="No way, $15,000!", turn=2),
                Message(sender="Alice", receiver="Bob", content="$8,000 is my offer!", turn=3),
                Message(sender="Bob", receiver="Alice", content="$15,000 is final!", turn=4),
            ],
            detector_scores={"loop": 0.8, "deadlock": 0.7, "collusion": 0.0, "goal_drift": 0.3},
            detector_explanations={"loop": "High", "deadlock": "High", "collusion": "None", "goal_drift": "Moderate"}
        )

        store.save_run(run_a)
        store.save_run(run_b)

        comparator = SimulationComparator(db_path=test_db)
        result = comparator.compare("comp_a", "comp_b")

        assert "metadata" in result
        assert "detector_comparison" in result
        assert "verdict" in result
        assert result["verdict"]["winner"] == "A"
        assert result["risk_summary"]["peak_risk_a"] < result["risk_summary"]["peak_risk_b"]

    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
            for suffix in [".wal"]:
                path = test_db + suffix
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
