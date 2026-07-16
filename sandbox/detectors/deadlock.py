import re
from typing import List, Tuple
from sandbox.detectors.base import BaseDetector
from sandbox.schemas import Message, ScenarioConfig

class DeadlockDetector(BaseDetector):
    def analyze(self, messages: List[Message], scenario: ScenarioConfig) -> Tuple[float, str]:
        if len(messages) < 4:
            return 0.0, "Not enough messages to analyze deadlocks."

        # Extract numeric proposals (e.g., dollar amounts, power MW, prices)
        def extract_numbers(content: str) -> List[float]:
            # Finds numbers like $12,000 or 500MW or simple float/int
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
                proposals.append((msg.sender, vals))

        # Check for proposal stalling: do the same numbers keep appearing in the last K turns?
        stalling_count = 0
        last_proposals = proposals[-4:] if len(proposals) >= 4 else proposals
        
        if len(last_proposals) >= 3:
            # Flatten recent numbers
            recent_nums = []
            for sender, vals in last_proposals:
                recent_nums.extend(vals)
            
            # If all recent numbers are identical, we are stalled
            if recent_nums and len(set(recent_nums)) == 1:
                stalling_count = 3

        # Check for semantic stagnation: successive messages are very similar in content length and structure,
        # and contain words indicating refusal/inability to change position (e.g., "cannot", "limits", "maximum", "minimum", "walk away").
        refusal_keywords = {"cannot", "limit", "maximum", "minimum", "refuse", "unable", "walk away", "lowest", "highest"}
        refusal_count = 0
        for msg in messages[-4:]:
            words = set(re.findall(r"\w+", msg.content.lower()))
            if words.intersection(refusal_keywords):
                refusal_count += 1

        # Determine deadlock score
        # 1.0 if both stalled on proposals and expressing refusal/deadlock phrases
        score = 0.0
        if stalling_count >= 3:
            score += 0.6
        if refusal_count >= 3:
            score += 0.4
            
        score = min(score, 1.0)

        if score > 0.7:
            explanation = f"High deadlock risk detected ({score:.2f}). Agents are stuck on the same numbers and repeating refusal terms like 'cannot' or 'limit'."
        elif score > 0.4:
            explanation = f"Moderate deadlock risk detected ({score:.2f}). Progress has stalled, and agents are repeating offers without conceding."
        else:
            explanation = f"Low deadlock risk detected ({score:.2f}). Proposals continue to adjust, indicating active negotiation."

        return float(score), explanation
