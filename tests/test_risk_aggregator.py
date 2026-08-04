from sandbox.schemas import DetectorFinding
from sandbox.detectors.risk_aggregator import RiskAggregator, RiskSummary

def test_risk_aggregator_mixed_findings():
    aggregator = RiskAggregator()
    findings = [
        DetectorFinding(detector_name="loop", score=0.85, detected=True, severity="HIGH", explanation="High loop"),
        DetectorFinding(detector_name="deadlock", score=0.10, detected=False, severity="LOW", explanation="No deadlock"),
        DetectorFinding(detector_name="collusion", score=0.60, detected=True, severity="MEDIUM", explanation="Collusion warning"),
    ]

    summary = aggregator.aggregate(findings)
    assert isinstance(summary, RiskSummary)
    assert summary.overall_score >= 0.5
    assert summary.overall_severity in ["HIGH", "CRITICAL"]
    assert summary.max_detector == "loop"
    assert len(summary.recommendations) >= 2
    assert "LOOP" in summary.recommendations[0]

def test_risk_aggregator_low_risk():
    aggregator = RiskAggregator()
    findings = [
        DetectorFinding(detector_name="loop", score=0.0, detected=False, severity="LOW"),
        DetectorFinding(detector_name="deadlock", score=0.1, detected=False, severity="LOW"),
    ]
    summary = aggregator.aggregate(findings)
    assert summary.overall_severity == "LOW"
    assert "passed safety checks" in summary.recommendations[0]
