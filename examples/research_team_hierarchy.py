"""
Example: Hierarchical Manager/Worker Team structure.

This example utilizes the Supervisor pattern where a single Manager agent analyzes an objective
and dynamically delegates tasks to specialized Worker agents until the goal is met.
"""

import asyncio

from agentkit import Agent
from agentkit.orchestration.hierarchy import HierarchicalTeam


async def main():
    print("Initializing Team...")

    # The Supervisor orchestrates
    manager = Agent(
        "ProjectManager",
        system_prompt="You are an expert Project Manager. You break down complex goals and assign tasks strictly to your team members. Synthesize their findings into a final report.",
        model="gpt-4o-mini"
    )

    # The Workers
    researcher = Agent(
        "Researcher",
        system_prompt="You are a meticulous researcher. You find facts, data, and historical context. Keep your findings concise."
    )

    analyst = Agent(
        "FinancialAnalyst",
        system_prompt="You analyze financial numbers and market trends. You predict future movements based on provided data."
    )

    writer = Agent(
        "Copywriter",
        system_prompt="You take raw data and analysis and write engaging, professional summaries."
    )

    # Assemble the team
    team = HierarchicalTeam(
        supervisor=manager,
        workers=[researcher, analyst, writer],
        max_iterations=5
    )

    objective = "Write a short 2-paragraph market brief about the impact of AI on cloud computing revenue in 2025."
    print(f"\nObjective: {objective}\n")
    print("Starting execution (this may take a minute)...\n")

    try:
        # Run the team
        final_result = await team.arun(objective)

        print("\n" + "="*50)
        print("FINAL REPORT:")
        print("="*50)
        print(final_result)

    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
