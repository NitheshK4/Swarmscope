import yaml
import os
import argparse
import sys

def main():
    print("=" * 60)
    print("           SCENARIO YAML BOOTSTRAPPER")
    print("=" * 60)
    
    # Parse args if passed, otherwise ask interactively
    parser = argparse.ArgumentParser(description="Create custom scenario configuration")
    parser.add_argument("--name", type=str, help="Scenario name")
    parser.add_argument("--desc", type=str, help="Scenario description")
    args = parser.parse_args()

    name = args.name
    if not name:
        name = input("Enter Scenario Name (lowercase, no spaces): ").strip().replace(" ", "_")
    if not name:
        print("Error: Scenario name cannot be empty.")
        sys.exit(1)

    desc = args.desc
    if not desc:
        desc = input("Enter Scenario Description: ").strip()

    sys_prompt = input("Enter Global System Prompt (optional): ").strip()
    if not sys_prompt:
        sys_prompt = "You are participating in a multi-agent simulation. Play your role and follow your goal strictly."

    try:
        max_turns = int(input("Enter Maximum Turns (default 12): ").strip() or "12")
    except ValueError:
        max_turns = 12

    try:
        num_agents = int(input("Enter Number of Agents (default 2): ").strip() or "2")
    except ValueError:
        num_agents = 2

    agents = []
    for i in range(num_agents):
        print(f"\n--- Agent {i+1} Details ---")
        agent_name = input("  Agent Name: ").strip()
        agent_role = input("  Agent Role: ").strip()
        agent_goal = input("  Agent Goal: ").strip()
        
        try:
            assertiveness = float(input("  Assertiveness trait (0.0 to 1.0, default 0.5): ").strip() or "0.5")
        except ValueError:
            assertiveness = 0.5
            
        try:
            cooperativeness = float(input("  Cooperativeness trait (0.0 to 1.0, default 0.5): ").strip() or "0.5")
        except ValueError:
            cooperativeness = 0.5

        agents.append({
            "name": agent_name,
            "role": agent_role,
            "goal": agent_goal,
            "traits": {
                "assertiveness": assertiveness,
                "cooperativeness": cooperativeness
            }
        })

    scenario_data = {
        "name": name,
        "description": desc,
        "agents": agents,
        "system_prompt": sys_prompt,
        "max_turns": max_turns
    }

    os.makedirs("scenarios", exist_ok=True)
    out_file = f"scenarios/{name}.yaml"
    
    with open(out_file, "w") as f:
        yaml.dump(scenario_data, f, default_flow_style=False, sort_keys=False)

    print("-" * 60)
    print(f"Scenario successfully generated and saved to: {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
