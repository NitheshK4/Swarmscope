import os
import pytest
from sandbox.analytics.batch_runner import BatchRunner

def test_batch_runner_execution_and_isolation():
    test_db = "test_batch_runner_isolation.duckdb"
    if os.path.exists(test_db):
        os.remove(test_db)

    try:
        scenario_path = "scenarios/negotiation.yaml"
        assert os.path.exists(scenario_path)

        runner = BatchRunner(
            scenario_path=scenario_path,
            backend_name="dummy",
            base_temperature=0.7,
            db_path=test_db,
            max_concurrency=2,
        )

        report = runner.run_batch(count=4)
        assert report["scenario_name"] == "negotiation"
        assert report["total_runs"] == 4
        assert report["completed_runs"] == 4
        assert report["failed_runs"] == 0
        assert "failure_rates" in report
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
            if os.path.exists(test_db + ".wal"):
                try:
                    os.remove(test_db + ".wal")
                except OSError:
                    pass
