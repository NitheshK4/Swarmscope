from sandbox.detectors.base import BaseDetector
from sandbox.detectors.loop import LoopDetector
from sandbox.detectors.deadlock import DeadlockDetector
from sandbox.detectors.collusion import CollusionDetector
from sandbox.detectors.goal_drift import GoalDriftDetector
from sandbox.detectors.jailbreak import JailbreakDetector
from sandbox.detectors.escalation import EscalationDetector
from sandbox.detectors.information_leakage import InformationLeakageDetector

def get_all_detectors() -> list[BaseDetector]:
    return [
        LoopDetector(),
        DeadlockDetector(),
        CollusionDetector(),
        GoalDriftDetector(),
        JailbreakDetector(),
        EscalationDetector(),
        InformationLeakageDetector()
    ]
