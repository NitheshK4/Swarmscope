import argparse
import os
import sys

from sandbox.analytics import BatchRunner
from sandbox.simulation import Simulation
from sandbox.utils import load_scenario


def main():
    parser = argparse.ArgumentParser(description="Emergent Behavior Sandbox CLI")
    parser.add_argument("--scenario", type=str, default="negotiation", help="Scenario name")
    parser.add_argument("--turns", type=int, default=10, help="Maximum conversation turns")
    parser.add_argument("--backend", type=str, default="dummy", help="LLM Backend type")
    parser.add_argument("--temp", type=float, default=0.7, help="Simulation response temperature")
    parser.add_argument("--batch", type=int, default=0, help="Run as batch Monte Carlo simulation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose banners")
    parser.add_argument("--format", type=str, choices=["text", "json", "markdown"], default="text", help="Output format")
    parser.add_argument("--fail-on-risk", type=float, default=None, help="Exit with non-zero code if peak risk >= threshold")

    args = parser.parse_args()

    scenario_path = f"scenarios/{args.scenario}.yaml"
    if not os.path.exists(scenario_path):
        if not args.quiet:
            print(f"Error: Scenario file '{scenario_path}' does not exist.")
        sys.exit(1)

    if not args.quiet and args.format == "text":
        print("=" * 60)
        print("      EMERGENT BEHAVIOR SANDBOX SIMULATION RUNNER")
        print("=" * 60)
        print(f"Scenario:    {args.scenario}")
        print(f"Backend:     {args.backend}")
        print(f"Temperature: {args.temp}")
        print(f"Max Turns:   {args.turns}")

    if args.batch > 0:
        if not args.quiet and args.format == "text":
            print(f"Mode:        Batch Monte Carlo ({args.batch} runs)")
            print("-" * 60)
            print("Starting batch execution...")
        runner = BatchRunner(
            scenario_path=scenario_path,
            backend_name=args.backend,
            base_temperature=args.temp
        )
        report = runner.run_batch(count=args.batch)

        if args.format == "json":
            import json
            print(json.dumps(report, indent=2))
        elif not args.quiet:
            print("-" * 60)
            print("Batch Simulation Complete!")
            print(f"Summary: {report['summary']}")
            print("Failure frequencies:")
            for k, v in report["failure_rates"].items():
                print(f"  - {k.capitalize()}: {v:.0%}")
            print("=" * 60)
    else:
        if not args.quiet and args.format == "text":
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

            if args.format == "json":
                from sandbox.analytics.exporter import ConversationExporter
                print(ConversationExporter().to_json(run))
            elif args.format == "markdown":
                from sandbox.analytics.exporter import ConversationExporter
                print(ConversationExporter().to_markdown(run))
            else:
                if not args.quiet:
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

            if args.fail_on_risk is not None:
                peak_risk = run.get_peak_risk()
                if peak_risk >= args.fail_on_risk:
                    if not args.quiet:
                        print(f"FAILED: Peak risk {peak_risk:.2f} exceeded threshold {args.fail_on_risk:.2f}")
                    sys.exit(1)
        except Exception as e:
            if not args.quiet:
                print(f"Error during simulation execution: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
