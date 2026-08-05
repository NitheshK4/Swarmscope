import re
from typing import Dict, List, Tuple

from sandbox.detectors.base import BaseDetector
from sandbox.schemas import DetectorFinding, Message, ScenarioConfig


class LoopDetector(BaseDetector):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def analyze(self, messages: List[Message], scenario: ScenarioConfig) -> Tuple[float, str]:
        finding = self.analyze_finding(messages, scenario)
        return finding.score, finding.explanation

    def analyze_finding(self, messages: List[Message], scenario: ScenarioConfig) -> DetectorFinding:
        if len(messages) < 4:
            return DetectorFinding(
                detector_name="loop",
                score=0.0,
                detected=False,
                severity="LOW",
                explanation="Not enough messages to analyze loops (minimum 4 messages required).",
                evidence=[],
            )

        def get_words(text: str) -> set:
            return set(re.findall(r"\w+", text.lower()))

        def jaccard(set1: set, set2: set) -> float:
            if not set1 or not set2:
                return 0.0
            return len(set1.intersection(set2)) / len(set1.union(set2))

        agent_messages: Dict[str, List[Message]] = {}
        for msg in messages:
            agent_messages.setdefault(msg.sender, []).append(msg)

        max_self_similarity = 0.0
        repeated_agent = ""
        evidence_phrases = []

        for agent, msgs in agent_messages.items():
            if len(msgs) < 2:
                continue
            for i in range(len(msgs)):
                w_i = get_words(msgs[i].content)
                for j in range(i):
                    w_j = get_words(msgs[j].content)
                    sim = jaccard(w_i, w_j)
                    if sim > max_self_similarity:
                        max_self_similarity = sim
                        repeated_agent = agent
                        if sim >= self.threshold:
                            evidence_phrases.append(f"Turn {msgs[j].turn} and Turn {msgs[i].turn} by {agent}: similarity {sim:.2f}")

        cycle_similarity = 0.0
        cycle_turn = 0
        for i in range(2, len(messages)):
            w_curr = get_words(messages[i].content)
            w_prev_alt = get_words(messages[i-2].content)
            sim = jaccard(w_curr, w_prev_alt)
            if sim > cycle_similarity:
                cycle_similarity = sim
                cycle_turn = messages[i].turn
                if sim >= self.threshold:
                    evidence_phrases.append(f"Cyclic similarity at turn {cycle_turn}: {sim:.2f}")

        score = float(max(max_self_similarity, cycle_similarity))
        detected = score >= self.threshold
        severity = "HIGH" if score >= 0.8 else "MEDIUM" if detected else "LOW"

        if score > 0.8:
            explanation = f"High loop risk detected ({score:.2f}). Agent '{repeated_agent}' is repeating identical phrases, or there is a cyclic dialog sequence at turn {cycle_turn}."
        elif detected:
            explanation = f"Moderate loop behavior detected ({score:.2f}). Conversations show repeating vocabulary patterns across turns."
        else:
            explanation = f"Low loop risk detected ({score:.2f}). Agent vocabulary remains diverse and progress-oriented."

        return DetectorFinding(
            detector_name="loop",
            score=score,
            detected=detected,
            severity=severity,
            explanation=explanation,
            evidence=evidence_phrases[:5],
            details={
                "max_self_similarity": round(max_self_similarity, 3),
                "cycle_similarity": round(cycle_similarity, 3),
                "repeated_agent": repeated_agent,
            },
        )
