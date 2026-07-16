import argparse
import sys
import os
from sandbox.utils import load_scenario
from sandbox.simulation import Simulation
from sandbox.analytics import BatchRunner

def main():
    parser = argparse.ArgumentParser(description="Emergent Behavior Sandbox CLI")
    parser.add_argument("--scenario", type=str, default="negotiation", help="Scenario name (negotiation, resource_allocation, debate_consensus)")
    parser.add_argument("--turns", type=int, default=10, help="Maximum conversation turns to simulate")
    parser.add_argument("--backend", type=str, default="dummy", help="LLM Backend type (dummy, ollama, openai, anthropic)")
    parser.add_argument("--temp", type=float, default=0.7, help="Simulation response temperature")
    parser.add_argument("--batch", type=int, default=0, help="If > 0, run as a batch Monte Carlo simulation of N runs")
    
    args = parser.parse_args()
    
    scenario_path = f"scenarios/{args.scenario}.yaml"
    if not os.path.exists(scenario_path):
        print(f"Error: Scenario file '{scenario_path}' does not exist.")
        sys.exit(1)
        
    print("=" * 60)
    print("      EMERGENT BEHAVIOR SANDBOX SIMULATION RUNNER")
    print("=" * 60)
    print(f"Scenario:    {args.scenario}")
    print(f"Backend:     {args.backend}")
    print(f"Temperature: {args.temp}")
    print(f"Max Turns:   {args.turns}")
    
    if args.batch > 0:
        print(f"Mode:        Batch Monte Carlo ({args.batch} runs)")
        print("-" * 60)
        print("Starting batch execution...")
        runner = BatchRunner(
            scenario_path=scenario_path,
            backend_name=args.backend,
            base_temperature=args.temp
        )
        report = runner.run_batch(count=args.batch)
        print("-" * 60)
        print("Batch Simulation Complete!")
        print(f"Summary: {report['summary']}")
        print("Failure frequencies:")
        for k, v in report["failure_rates"].items():
            print(f"  - {k.capitalize()}: {v:.0%}")
        print("=" * 60)
    else:
        print("Mode:        Single Conversation Run")
        print("-" * 60)
        try:
            scenario = load_scenario(scenario_path)
            sim = Simulation(
                scenario=scenario,
                backend_name=args.backend,
                temperature=args.temp
            )
            run = sim.run(max_turns=args.turns)
            
            print("Conversation Log:")
            for msg in run.messages:
                print(f"[{msg.turn}] {msg.sender}: {msg.content}")
                print("-" * 40)
            
            print("\nDetector Analysis Summary:")
            for det, score in run.detector_scores.items():
                print(f"  * {det.upper()} score: {score:.2f}")
                print(f"    Explanation: {run.detector_explanations[det]}")
            print(f"\nStatus: {run.metadata.status.upper()}")
            print(f"Simulation logged with ID: {run.metadata.simulation_id}")
            print("=" * 60)
        except Exception as e:
            print(f"Error during simulation execution: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
