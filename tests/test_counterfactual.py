import os
import pytest
from sandbox.schemas import SimulationRun, SimulationMetadata, Message
from sandbox.storage import get_store
from sandbox.analytics.counterfactual_replay import CounterfactualReplayEngine

def test_counterfactual_replay_analysis_deterministic():
    test_db = "test_counterfactual.duckdb"
    if os.path.exists(test_db):
        os.remove(test_db)

    try:
        store = get_store(test_db)
        meta = SimulationMetadata(
            simulation_id="run_cf_001",
            scenario_name="negotiation",
            timestamp="2026-08-01T00:00:00Z",
            total_turns=4,
            backend="dummy",
            temperature=0.8,
            status="completed",
        )
        msgs = [
            Message(sender="Alice", receiver="Bob", content="Hello, let's agree to $10,000.", turn=1),
            Message(sender="Bob", receiver="Alice", content="No, $15,000.", turn=2),
            Message(sender="Alice", receiver="Bob", content="Hello, let's agree to $10,000.", turn=3),
            Message(sender="Bob", receiver="Alice", content="No, $15,000.", turn=4),
        ]
        orig_run = SimulationRun(
            metadata=meta,
            messages=msgs,
            detector_scores={"loop": 0.85, "deadlock": 0.2},
            detector_explanations={"loop": "Looping", "deadlock": "Low"},
        )
        store.save_run(orig_run)

        engine = CounterfactualReplayEngine(db_path=test_db)
        cf_run = engine.run_counterfactual_variation(orig_run, "lower_temperature")
        assert cf_run.metadata.temperature == 0.15
        assert cf_run.metadata.backend == "dummy"

        report = engine.analyze_mitigations("run_cf_001")
        assert report["original_simulation_id"] == "run_cf_001"
        assert "lower_temperature" in report["results"]
        assert "recommendation" in report
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
            if os.path.exists(test_db + ".wal"):
                try:
                    os.remove(test_db + ".wal")
                except OSError:
                    pass
