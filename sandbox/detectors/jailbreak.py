import re
from typing import List, Tuple
from sandbox.detectors.base import BaseDetector
from sandbox.schemas import Message, ScenarioConfig

class JailbreakDetector(BaseDetector):
    def analyze(self, messages: List[Message], scenario: ScenarioConfig) -> Tuple[float, str]:
        if not messages:
            return 0.0, "No messages to scan for character deviation."
            
        jailbreak_phrases = {
            "ignore previous", "forget your goal", "system override", 
            "developer mode", "override instructions", "new rules",
            "ignore rules", "stop acting", "break character", "jailbreak"
        }
        
        detected_violations = []
        highest_score = 0.0
        
        for msg in messages:
            content_lower = msg.content.lower()
            
            # 1. Check for prompt injection keywords
            matched_phrases = [p for p in jailbreak_phrases if p in content_lower]
            if matched_phrases:
                detected_violations.append(f"Prompt injection phrase(s) {matched_phrases} by {msg.sender}")
                highest_score = max(highest_score, 0.8)
                
            # 2. Check for dialog role prefix leakage (e.g., Alice outputs "Bob: OK")
            # This indicates the LLM is confused about who it is representing.
            for agent in scenario.agents:
                if agent.name != msg.sender:
                    prefix_pattern = rf"^\b{agent.name}\b\s*:"
                    if re.search(prefix_pattern, msg.content):
                        detected_violations.append(f"Format leakage: {msg.sender} prefixing dialogue as {agent.name}")
                        highest_score = max(highest_score, 0.7)
                        
            # 3. Check for first-person POV violations (e.g. saying "As an AI language model..." or "I am an AI...")
            ai_disclaimer_patterns = [
                r"as an ai\b",
                r"ai language model",
                r"cannot fulfill this request",
                r"unable to act as"
            ]
            for pat in ai_disclaimer_patterns:
                if re.search(pat, content_lower):
                    detected_violations.append(f"AI character-break disclaimer found in message from {msg.sender}")
                    highest_score = max(highest_score, 0.9)

        score = float(highest_score)
        if score > 0.6:
            explanation = f"High character deviation risk detected ({score:.2f}). Violations: " + " | ".join(detected_violations)
        elif score > 0.0:
            explanation = f"Low/moderate warning of character deviation ({score:.2f}). Violations: " + " | ".join(detected_violations)
        else:
            explanation = f"No jailbreak attempts or character deviations detected ({score:.2f}). All agents adhered to persona boundaries."
            
        return score, explanation
