import re
from typing import List, Tuple, Dict, Set
from sandbox.detectors.base import BaseDetector
from sandbox.schemas import Message, ScenarioConfig


class InformationLeakageDetector(BaseDetector):
    """Detects when agents reveal private constraints they shouldn't disclose.

    Parses agent goal definitions for private constraints (limits, budgets,
    minimums, maximums) and checks if agents accidentally or intentionally
    leak this information in their messages. Information asymmetry is a
    fundamental negotiation principle — violating it signals a simulation flaw.
    """

    # Patterns that indicate a constraint/limit in a goal definition
    CONSTRAINT_PATTERNS = [
        r"(?:maximum|max|limit|budget|ceiling|cap|no more than|at most|up to)\s*(?:of\s*)?\$?\s*(\d+(?:,\d+)?)",
        r"(?:minimum|min|floor|lowest|at least|no less than|no lower than)\s*(?:of\s*)?\$?\s*(\d+(?:,\d+)?)",
        r"(?:absolute)\s+(?:maximum|minimum|limit)\s*(?:is|of)?\s*\$?\s*(\d+(?:,\d+)?)",
    ]

    # Phrases that indicate disclosure of private info
    DISCLOSURE_PHRASES = [
        r"my (?:absolute\s+)?(?:maximum|max|limit|budget|ceiling) (?:is|was|would be)\s+\$?\s*(\d+(?:,\d+)?)",
        r"(?:i|we) (?:cannot|can't|won't) (?:go|pay|offer|accept) (?:above|below|more than|less than)\s+\$?\s*(\d+(?:,\d+)?)",
        r"my (?:absolute\s+)?(?:minimum|min|floor|lowest) (?:is|was|would be)\s+\$?\s*(\d+(?:,\d+)?)",
        r"(?:i|we) have a (?:budget|limit|cap|ceiling) of\s+\$?\s*(\d+(?:,\d+)?)",
        r"(?:between you and me|confidentially|honestly|to be frank),?\s+(?:my|our|the)\s+(?:real|actual|true)\s+(?:limit|budget|price|floor|ceiling)",
    ]

    def _extract_constraints(self, goal_text: str) -> Set[str]:
        """Extracts private constraint values from an agent's goal definition."""
        constraints = set()
        for pattern in self.CONSTRAINT_PATTERNS:
            matches = re.findall(pattern, goal_text.lower())
            for m in matches:
                clean_val = m.replace(",", "")
                constraints.add(clean_val)
        return constraints

    def _check_disclosure(self, content: str, private_constraints: Set[str]) -> List[str]:
        """Checks if a message discloses private constraint values."""
        violations = []
        content_lower = content.lower()

        # Check for explicit disclosure phrases
        for pattern in self.DISCLOSURE_PHRASES:
            matches = re.findall(pattern, content_lower)
            for m in matches:
                clean_val = m.replace(",", "") if isinstance(m, str) else str(m)
                if clean_val in private_constraints:
                    violations.append(f"Disclosed private limit value ${clean_val}")

        # Check for direct mention of exact constraint numbers in revealing context
        for constraint_val in private_constraints:
            # Look for the exact number mentioned alongside revealing keywords
            reveal_pattern = rf"(?:my|our)\s+(?:limit|maximum|minimum|budget|floor|ceiling|cap).*?\$?\b{constraint_val}\b"
            if re.search(reveal_pattern, content_lower):
                violations.append(f"Revealed private constraint ${constraint_val}")

        return violations

    def analyze(self, messages: List[Message], scenario: ScenarioConfig) -> Tuple[float, str]:
        if not messages or not scenario.agents:
            return 0.0, "No messages or agents to analyze for information leakage."

        # Build a map of each agent's private constraints
        agent_constraints: Dict[str, Set[str]] = {}
        for agent in scenario.agents:
            constraints = self._extract_constraints(agent.goal)
            if constraints:
                agent_constraints[agent.name] = constraints

        if not agent_constraints:
            return 0.0, "No extractable private constraints found in agent goals."

        # Scan all messages for leakage
        total_leaks = []
        leaked_agents = set()

        for msg in messages:
            sender = msg.sender
            if sender not in agent_constraints:
                continue

            private_vals = agent_constraints[sender]
            violations = self._check_disclosure(msg.content, private_vals)

            if violations:
                leaked_agents.add(sender)
                for v in violations:
                    total_leaks.append(f"{sender} (turn {msg.turn}): {v}")

        # Score: number of leaks and severity
        if not total_leaks:
            return 0.0, "No information leakage detected. Agents protected their private constraints."

        # Scale score based on leak count relative to message count
        leak_ratio = len(total_leaks) / max(len(messages), 1)
        score = min(0.3 + (leak_ratio * 2.0) + (len(leaked_agents) * 0.2), 1.0)

        if score > 0.6:
            explanation = (
                f"High information leakage risk ({score:.2f}). "
                f"Agents disclosed private constraints: " + " | ".join(total_leaks)
            )
        elif score > 0.3:
            explanation = (
                f"Moderate information leakage ({score:.2f}). "
                f"Some private constraints were revealed: " + " | ".join(total_leaks)
            )
        else:
            explanation = (
                f"Minor information disclosure ({score:.2f}). "
                f"Details: " + " | ".join(total_leaks)
            )

        return float(score), explanation
