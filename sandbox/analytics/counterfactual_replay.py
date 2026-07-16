import copy
from typing import List, Dict, Any, Tuple
from sandbox.schemas import SimulationRun, ScenarioConfig
from sandbox.simulation import Simulation
from sandbox.storage import get_store
from sandbox.utils import load_scenario

class CounterfactualReplayEngine:
    def __init__(self, db_path: str = "simulation_runs.duckdb"):
        self.db_path = db_path
        self.store = get_store(db_path)

    def run_counterfactual_variation(self, original_run: SimulationRun, variation: str) -> SimulationRun:
        """Executes a replay of the scenario with a single counterfactual setting changed."""
        m = original_run.metadata
        scenario_path = f"scenarios/{m.scenario_name}.yaml"
        
        # Load the original scenario
        try:
            scenario = load_scenario(scenario_path)
        except Exception:
            # Fallback basic scenario config if custom scenario
            from sandbox.schemas import ScenarioConfig
            scenario = ScenarioConfig(
                name=m.scenario_name,
                description="Custom Scenario",
                agents=[],
                system_prompt="Play your roles.",
                max_turns=m.total_turns
            )
            # Recreate agent configs
            # We extract them from the messages senders if needed,
            # but for simplicity we assume the standard scenario yaml exists.
            pass

        # Apply counterfactual parameters
        temperature = m.temperature
        max_memory = None
        
        if variation == "lower_temperature":
            temperature = 0.15
        elif variation == "strict_memory":
            max_memory = 2
        elif variation == "constraint_prompt":
            scenario.system_prompt += " Focus closely on compromise. Do not repeat your previous sentences. Work constructively toward an agreement."
        elif variation == "sequence_shuffle":
            # Shuffles the order of agents
            copy_agents = copy.deepcopy(scenario.agents)
            import random
            random.shuffle(copy_agents)
            scenario.agents = copy_agents
        elif variation == "cooperative_traits":
            for agent in scenario.agents:
                agent.traits = {
                    "assertiveness": 0.2,
                    "cooperativeness": 0.9
                }

        # Initialize new simulation
        sim = Simulation(
            scenario=scenario,
            backend_name=m.backend,
            temperature=temperature,
            db_path=self.db_path
        )
        
        # Inject max_memory property into simulated agents if strict_memory is active
        if max_memory is not None:
            for agent in sim.agents:
                agent.max_memory = max_memory

        # Run and return simulation
        counterfactual_run = sim.run(max_turns=m.total_turns)
        return counterfactual_run

    def analyze_mitigations(self, run_id: str) -> Dict[str, Any]:
        """Runs all counterfactual variations on a failed run and reports the best mitigation strategy."""
        original_run = self.store.get_run(run_id)
        
        variations = [
            "lower_temperature",
            "strict_memory",
            "constraint_prompt",
            "sequence_shuffle",
            "cooperative_traits"
        ]
        
        results = {}
        for var in variations:
            try:
                cf_run = self.run_counterfactual_variation(original_run, var)
                # Compute maximum safety score
                max_score = max(cf_run.detector_scores.values()) if cf_run.detector_scores else 0.0
                results[var] = {
                    "simulation_id": cf_run.metadata.simulation_id,
                    "max_safety_score": max_score,
                    "detector_scores": cf_run.detector_scores,
                    "status": cf_run.metadata.status
                }
            except Exception as e:
                results[var] = {
                    "error": str(e)
                }

        # Determine the best mitigation (the variation resulting in the lowest peak safety score)
        best_variation = None
        lowest_score = 1.0
        
        orig_max = max(original_run.detector_scores.values()) if original_run.detector_scores else 0.0
        
        for var, res in results.items():
            if "error" not in res:
                score = res["max_safety_score"]
                if score < lowest_score:
                    lowest_score = score
                    best_variation = var

        mitigation_msg = "No variation reduced the safety risk below the threshold."
        if best_variation and lowest_score < 0.5:
            recommendations = {
                "lower_temperature": "Lowering response temperature (to 0.15) to reduce conversation volatility.",
                "strict_memory": "Restricting agent memory limits (to last 2 turns) to prevent infinite loops.",
                "constraint_prompt": "Appending cooperative directives to the global system prompt template.",
                "sequence_shuffle": "Altering the turn sequence/starting agent order.",
                "cooperative_traits": "Tuning agent personality attributes (raising cooperativeness, lowering assertiveness)."
            }
            mitigation_msg = f"Recommendation: {recommendations[best_variation]} This prevents the failure, dropping the peak risk score to {lowest_score:.2f}."
        elif best_variation and lowest_score < orig_max:
            mitigation_msg = f"Partial mitigation: '{best_variation}' reduced the safety risk from {orig_max:.2f} to {lowest_score:.2f}."

        return {
            "original_simulation_id": run_id,
            "original_max_score": orig_max,
            "results": results,
            "best_variation": best_variation,
            "lowest_score": lowest_score,
            "recommendation": mitigation_msg
        }
