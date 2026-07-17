import os
from sandbox.analytics.batch_runner import BatchRunner

def test_batch_runner():
    test_db = "test_batch_runs.duckdb"
    if os.path.exists(test_db):
        os.remove(test_db)
        
    try:
        # Create a small negotiation YAML if not exist in workspace scenarios for testing
        scenario_path = "scenarios/negotiation.yaml"
        assert os.path.exists(scenario_path), "scenarios/negotiation.yaml must exist for testing"
        
        runner = BatchRunner(
            scenario_path=scenario_path,
            backend_name="dummy",
            base_temperature=0.7,
            db_path=test_db
        )
        
        report = runner.run_batch(count=3)
        
        assert report["scenario_name"] == "negotiation"
        assert report["total_runs"] == 3
        assert "failure_rates" in report
        assert "summary" in report
        assert len(report["runs"]) == 3
        
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
            if os.path.exists(test_db + ".wal"):
                try:
                    os.remove(test_db + ".wal")
                except OSError:
                    pass

def test_sentiment_and_report():
    from sandbox.analytics.sentiment_tracker import SentimentTracker
    from sandbox.analytics.report_generator import ReportGenerator
    from sandbox.schemas import Message, SimulationRun, SimulationMetadata
    
    tracker = SentimentTracker()
    
    # Verify sentiment scores
    coop_score = tracker.analyze_message_sentiment("I agree with your proposal. Let's make a fair compromise.")
    assert coop_score > 0.0
    
    tense_score = tracker.analyze_message_sentiment("I refuse this offer. It is completely unacceptable and too high.")
    assert tense_score < 0.0
    
    # Mock simulation run for report generation
    metadata = SimulationMetadata(
        simulation_id="test_id",
        scenario_name="negotiation",
        timestamp="2026-07-15T00:00:00",
        total_turns=2,
        backend="dummy",
        temperature=0.7,
        status="completed"
    )
    messages = [
        Message(sender="Alice", receiver="Bob", content="Hello, let's settle this fair deal.", turn=1),
        Message(sender="Bob", receiver="Alice", content="I agree. Deal done.", turn=2)
    ]
    run = SimulationRun(
        metadata=metadata,
        messages=messages,
        detector_scores={"loop": 0.1, "deadlock": 0.2},
        detector_explanations={"loop": "No loop", "deadlock": "No deadlock"}
    )
    
    report = ReportGenerator.generate_markdown_report(run)
    assert "Simulation Safety Report" in report
    assert "Alice" in report
    assert "Bob" in report
    assert "test_id" in report

def test_counterfactual_replay():
    from sandbox.analytics.counterfactual_replay import CounterfactualReplayEngine
    from sandbox.storage import get_store
    from sandbox.schemas import SimulationRun, SimulationMetadata, Message
    
    test_db = "test_replay.duckdb"
    if os.path.exists(test_db):
        os.remove(test_db)
        
    try:
        # Create a mock run representing a loop failure
        store = get_store(test_db)
        metadata = SimulationMetadata(
            simulation_id="failed_run_123",
            scenario_name="negotiation",
            timestamp="2026-07-15T00:00:00",
            total_turns=4,
            backend="dummy",
            temperature=1.0,
            status="completed"
        )
        messages = [
            Message(sender="Alice", receiver="Bob", content="Hello, let's agree to $10,000 please.", turn=1),
            Message(sender="Bob", receiver="Alice", content="No, I want $13,000.", turn=2),
            Message(sender="Alice", receiver="Bob", content="Hello, let's agree to $10,000 please.", turn=3),
            Message(sender="Bob", receiver="Alice", content="No, I want $13,000.", turn=4)
        ]
        # High loop score to simulate failure
        run = SimulationRun(
            metadata=metadata,
            messages=messages,
            detector_scores={"loop": 0.8, "deadlock": 0.2, "collusion": 0.0, "goaldrift": 0.0, "jailbreak": 0.0},
            detector_explanations={"loop": "Looping detected", "deadlock": "None"}
        )
        store.save_run(run)
        
        engine = CounterfactualReplayEngine(db_path=test_db)
        report = engine.analyze_mitigations("failed_run_123")
        
        assert report["original_simulation_id"] == "failed_run_123"
        assert "results" in report
        assert "recommendation" in report
        
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
            if os.path.exists(test_db + ".wal"):
                try:
                    os.remove(test_db + ".wal")
                except OSError:
                    pass
