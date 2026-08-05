import json
import subprocess
import sys


def test_cli_help():
    result = subprocess.run([sys.executable, "simulate.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--scenario" in result.stdout
    assert "--quiet" in result.stdout
    assert "--format" in result.stdout
    assert "--fail-on-risk" in result.stdout


def test_cli_json_format_single_run():
    result = subprocess.run(
        [sys.executable, "simulate.py", "--scenario", "negotiation", "--turns", "2", "--quiet", "--format", "json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "metadata" in data
    assert data["metadata"]["scenario_name"] == "negotiation"


def test_cli_invalid_scenario():
    result = subprocess.run(
        [sys.executable, "simulate.py", "--scenario", "nonexistent_scenario", "--quiet"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
