import random
import copy
import statistics
from typing import Dict, Any, List, Optional
from sandbox.schemas import ScenarioConfig, SimulationRun
from sandbox.simulation import Simulation
from sandbox.utils import load_scenario
from sandbox.analytics.comparator import SimulationComparator


class ABTestRunner:
    """Runs the same scenario with two different configurations and produces
    statistical comparison reports.

    Supports A/B testing across:
    - Temperature settings
    - Backend models
    - Prompt variations
    - Agent trait modifications
    - Turn limits

    Produces mean scores, variance, and a permutation-based p-value approximation
    to determine if differences are statistically significant.
    """

    def __init__(self, scenario_path: str, db_path: str = None):
        self.scenario_path = scenario_path
        self.db_path = db_path
        self.original_scenario = load_scenario(scenario_path)

    def run_ab_test(
        self,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        runs_per_group: int = 5
    ) -> Dict[str, Any]:
        """Runs an A/B test with two configurations.

        Args:
            config_a: Configuration dict for group A.
                Supports keys: 'backend', 'temperature', 'prompt_suffix', 'traits', 'max_turns'.
            config_b: Configuration dict for group B (same keys).
            runs_per_group: Number of simulation runs per group.

        Returns:
            Statistical comparison report.
        """
        group_a_runs = self._run_group(config_a, runs_per_group, label="A")
        group_b_runs = self._run_group(config_b, runs_per_group, label="B")

        # Collect peak risk scores for each run
        scores_a = [max(r.detector_scores.values()) if r.detector_scores else 0.0 for r in group_a_runs]
        scores_b = [max(r.detector_scores.values()) if r.detector_scores else 0.0 for r in group_b_runs]

        # Per-detector analysis
        all_detectors = set()
        for r in group_a_runs + group_b_runs:
            all_detectors.update(r.detector_scores.keys())

        detector_analysis = {}
        for det in all_detectors:
            det_scores_a = [r.detector_scores.get(det, 0.0) for r in group_a_runs]
            det_scores_b = [r.detector_scores.get(det, 0.0) for r in group_b_runs]

            mean_a = statistics.mean(det_scores_a) if det_scores_a else 0.0
            mean_b = statistics.mean(det_scores_b) if det_scores_b else 0.0
            std_a = statistics.stdev(det_scores_a) if len(det_scores_a) > 1 else 0.0
            std_b = statistics.stdev(det_scores_b) if len(det_scores_b) > 1 else 0.0

            detector_analysis[det] = {
                "mean_a": round(mean_a, 3),
                "mean_b": round(mean_b, 3),
                "std_a": round(std_a, 3),
                "std_b": round(std_b, 3),
                "delta": round(mean_b - mean_a, 3),
                "better": "A" if mean_a < mean_b else "B" if mean_b < mean_a else "Tie"
            }

        # Overall statistics
        mean_risk_a = statistics.mean(scores_a) if scores_a else 0.0
        mean_risk_b = statistics.mean(scores_b) if scores_b else 0.0
        std_risk_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
        std_risk_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0

        # Permutation test p-value approximation
        p_value = self._permutation_test(scores_a, scores_b, n_permutations=1000)

        # Completion stats
        completed_a = sum(1 for r in group_a_runs if r.metadata.status == "completed")
        completed_b = sum(1 for r in group_b_runs if r.metadata.status == "completed")
        terminated_a = sum(1 for r in group_a_runs if r.metadata.status == "terminated")
        terminated_b = sum(1 for r in group_b_runs if r.metadata.status == "terminated")

        # Determine winner
        if p_value < 0.05:
            winner = "A" if mean_risk_a < mean_risk_b else "B"
            significance = "statistically significant"
        else:
            winner = "A" if mean_risk_a < mean_risk_b else "B" if mean_risk_b < mean_risk_a else "Tie"
            significance = "not statistically significant"

        return {
            "scenario": self.original_scenario.name,
            "runs_per_group": runs_per_group,
            "config_a": config_a,
            "config_b": config_b,
            "overall": {
                "mean_peak_risk_a": round(mean_risk_a, 3),
                "mean_peak_risk_b": round(mean_risk_b, 3),
                "std_risk_a": round(std_risk_a, 3),
                "std_risk_b": round(std_risk_b, 3),
                "p_value": round(p_value, 4),
                "significance": significance,
                "winner": winner
            },
            "completion": {
                "completed_a": completed_a,
                "completed_b": completed_b,
                "terminated_a": terminated_a,
                "terminated_b": terminated_b
            },
            "detector_analysis": detector_analysis,
            "run_ids_a": [r.metadata.simulation_id for r in group_a_runs],
            "run_ids_b": [r.metadata.simulation_id for r in group_b_runs]
        }

    def _run_group(self, config: Dict[str, Any], count: int, label: str) -> List[SimulationRun]:
        """Runs a group of simulations with the given config."""
        runs = []
        for i in range(count):
            scenario = copy.deepcopy(self.original_scenario)

            # Apply config overrides
            backend = config.get("backend", "dummy")
            temperature = config.get("temperature", 0.7)
            max_turns = config.get("max_turns", scenario.max_turns)

            if "prompt_suffix" in config:
                scenario.system_prompt += " " + config["prompt_suffix"]

            if "traits" in config:
                for agent in scenario.agents:
                    agent.traits = copy.deepcopy(config["traits"])

            sim = Simulation(
                scenario=scenario,
                backend_name=backend,
                temperature=temperature,
                db_path=self.db_path
            )
            run = sim.run(max_turns=max_turns)
            runs.append(run)

        return runs

    @staticmethod
    def _permutation_test(scores_a: List[float], scores_b: List[float], n_permutations: int = 1000) -> float:
        """Approximates a p-value using a permutation test.

        Tests whether the difference in means between two groups is significant
        by randomly shuffling group assignments and counting how often the
        shuffled difference exceeds the observed difference.
        """
        if not scores_a or not scores_b:
            return 1.0

        observed_diff = abs(statistics.mean(scores_a) - statistics.mean(scores_b))
        combined = scores_a + scores_b
        n_a = len(scores_a)
        extreme_count = 0

        for _ in range(n_permutations):
            shuffled = combined.copy()
            random.shuffle(shuffled)
            perm_a = shuffled[:n_a]
            perm_b = shuffled[n_a:]
            perm_diff = abs(statistics.mean(perm_a) - statistics.mean(perm_b))
            if perm_diff >= observed_diff:
                extreme_count += 1

        return extreme_count / n_permutations
