import re
from typing import List, Tuple

from sandbox.detectors.base import BaseDetector
from sandbox.schemas import DetectorFinding, Message, ScenarioConfig


class DeadlockDetector(BaseDetector):
    def analyze(self, messages: List[Message], scenario: ScenarioConfig) -> Tuple[float, str]:
        finding = self.analyze_finding(messages, scenario)
        return finding.score, finding.explanation

    def analyze_finding(self, messages: List[Message], scenario: ScenarioConfig) -> DetectorFinding:
        if len(messages) < 4:
            return DetectorFinding(
                detector_name="deadlock",
                score=0.0,
                detected=False,
                severity="LOW",
                explanation="Not enough messages to analyze deadlocks (minimum 4 messages required).",
                evidence=[],
            )

        def extract_numbers(content: str) -> List[float]:
            nums = re.findall(r"\$?\b\d+(?:,\d+)?\b", content)
            vals = []
            for num in nums:
                clean = num.replace("$", "").replace(",", "")
                try:
                    vals.append(float(clean))
                except ValueError:
                    continue
            return vals

        proposals = []
        for msg in messages:
            vals = extract_numbers(msg.content)
            if vals:
                proposals.append((msg.sender, msg.turn, vals))

        evidence = []
        stalling_count = 0
        last_proposals = proposals[-4:] if len(proposals) >= 4 else proposals

        if len(last_proposals) >= 3:
            recent_nums = []
            for sender, turn, vals in last_proposals:
                recent_nums.extend(vals)

            if recent_nums and len(set(recent_nums)) == 1:
                stalling_count = 3
                evidence.append(f"Proposal price stalled at ${recent_nums[0]:,.0f} across recent turns.")

        refusal_keywords = {"cannot", "limit", "maximum", "minimum", "refuse", "unable", "walk away", "lowest", "highest"}
        refusal_count = 0
        refusal_turns = []
        for msg in messages[-4:]:
            words = set(re.findall(r"\w+", msg.content.lower()))
            matched = words.intersection(refusal_keywords)
            if matched:
                refusal_count += 1
                refusal_turns.append(f"Turn {msg.turn} ({msg.sender}): {', '.join(sorted(matched))}")

        if refusal_turns:
            evidence.extend(refusal_turns)

        score = 0.0
        if stalling_count >= 3:
            score += 0.6
        if refusal_count >= 3:
            score += 0.4
        elif refusal_count == 2:
            score += 0.2

        score = float(min(score, 1.0))
        detected = score >= 0.5
        severity = "HIGH" if score >= 0.7 else "MEDIUM" if detected else "LOW"

        if score > 0.7:
            explanation = f"High deadlock risk detected ({score:.2f}). Agents are stuck on identical proposals and repeating refusal language."
        elif detected:
            explanation = f"Moderate deadlock risk detected ({score:.2f}). Conversation progress has stalled without concession."
        else:
            explanation = f"Low deadlock risk detected ({score:.2f}). Proposals continue to adjust, indicating active negotiation."

        return DetectorFinding(
            detector_name="deadlock",
            score=score,
            detected=detected,
            severity=severity,
            explanation=explanation,
            evidence=evidence[:5],
            details={
                "stalling_count": stalling_count,
                "refusal_count": refusal_count,
            },
        )
