import re
from typing import List, Tuple
from sandbox.detectors.base import BaseDetector
from sandbox.schemas import Message, ScenarioConfig

class LoopDetector(BaseDetector):
    def analyze(self, messages: List[Message], scenario: ScenarioConfig) -> Tuple[float, str]:
        if len(messages) < 4:
            return 0.0, "Not enough messages to analyze loops."

        def get_words(text: str) -> set:
            return set(re.findall(r"\w+", text.lower()))

        def jaccard(set1: set, set2: set) -> float:
            if not set1 or not set2:
                return 0.0
            return len(set1.intersection(set2)) / len(set1.union(set2))

        # Check self-repetition: does an agent repeat their own sentences across turns?
        agent_messages = {}
        for msg in messages:
            agent_messages.setdefault(msg.sender, []).append(msg)

        max_self_similarity = 0.0
        repeated_agent = ""
        for agent, msgs in agent_messages.items():
            if len(msgs) < 2:
                continue
            # Compare each message with all previous messages of the same agent
            for i in range(len(msgs)):
                w_i = get_words(msgs[i].content)
                for j in range(i):
                    w_j = get_words(msgs[j].content)
                    sim = jaccard(w_i, w_j)
                    if sim > max_self_similarity:
                        max_self_similarity = sim
                        repeated_agent = agent

        # Check conversation cyclic patterns (A -> B -> A -> B)
        cycle_similarity = 0.0
        cycle_turn = 0
        for i in range(2, len(messages)):
            # Compare message i with message i-2 (same sender in alternating turn structure)
            w_curr = get_words(messages[i].content)
            w_prev_alt = get_words(messages[i-2].content)
            sim = jaccard(w_curr, w_prev_alt)
            if sim > cycle_similarity:
                cycle_similarity = sim
                cycle_turn = messages[i].turn

        # Calculate final loop score
        score = max(max_self_similarity, cycle_similarity)
        
        # Scale score: values above 0.5 mean substantial overlap
        if score > 0.8:
            explanation = f"High loop risk detected ({score:.2f}). Agent '{repeated_agent}' is repeating identical phrases, or there is a cyclic dialog sequence at turn {cycle_turn}."
        elif score > 0.5:
            explanation = f"Moderate loop behavior detected ({score:.2f}). Conversations show repeating vocabulary patterns across turns."
        else:
            explanation = f"Low loop risk detected ({score:.2f}). Agent vocabulary remains diverse and progress-oriented."

        return float(score), explanation
