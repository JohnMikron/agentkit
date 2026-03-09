#!/usr/bin/env python3
"""
Multi-agent orchestration example.

This example demonstrates:
- Creating a team of specialized agents
- Sequential and parallel execution strategies
- Workflows with conditional transitions
- Routing requests to specialized agents
"""

import asyncio

from agentkit import Agent
from agentkit.orchestration import Team, TeamConfig, TeamStrategy
from agentkit.orchestration.router import Router, RouteStrategy
from agentkit.orchestration.workflow import TransitionType, Workflow


async def main():
    """Run multi-agent orchestration examples."""

    # ============================================================
    # Example 1: Team with Sequential Strategy
    # ============================================================
    print("=" * 60)
    print("Example 1: Team - Sequential Execution")
    print("=" * 60)

    # Create specialized agents
    researcher = Agent(
        "researcher",
        model="gpt-4o-mini",
        system_prompt="You are a research specialist. Gather information concisely.",
    )

    writer = Agent(
        "writer",
        model="gpt-4o-mini",
        system_prompt="You are a content writer. Write engaging content.",
    )

    reviewer = Agent(
        "reviewer",
        model="gpt-4o-mini",
        system_prompt="You are an editor. Review and improve content.",
    )

    # Create team
    team = Team("content_team", strategy=TeamStrategy.SEQUENTIAL)
    team.add_agent(researcher)
    team.add_agent(writer)
    team.add_agent(reviewer)

    print(f"\nTeam created: {team}")
    print(f"Strategy: {team.config.strategy.value}")
    print(f"Agents: {[a.name for a in team.get_agents()]}")

    # ============================================================
    # Example 2: Team with Parallel Strategy
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 2: Team - Parallel Execution")
    print("=" * 60)

    parallel_team = Team(
        "parallel_team",
        config=TeamConfig(
            name="analysts",
            strategy=TeamStrategy.PARALLEL,
            max_parallel=3,
        ),
    )

    # Add multiple analysts
    for i in range(3):
        analyst = Agent(
            f"analyst_{i+1}",
            model="gpt-4o-mini",
            system_prompt=f"You are analyst #{i+1}. Provide unique insights.",
        )
        parallel_team.add_agent(analyst)

    print(f"\nParallel team: {parallel_team}")

    # ============================================================
    # Example 3: Workflow with Conditional Transitions
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 3: Workflow - State Machine")
    print("=" * 60)

    # Create workflow
    workflow = Workflow("research_workflow")

    # Add steps
    workflow.add_step(
        name="research",
        agent=researcher,
        prompt_template="Research this topic: {{ topic }}",
    )

    workflow.add_step(
        name="write",
        agent=writer,
        prompt_template="Write an article based on: {{ research_result }}",
    )

    workflow.add_step(
        name="review",
        agent=reviewer,
        prompt_template="Review and improve: {{ write_result }}",
    )

    # Add transitions
    workflow.add_transition("research", "write", TransitionType.ON_SUCCESS)
    workflow.add_transition("write", "review", TransitionType.ON_SUCCESS)

    print(f"\nWorkflow created: {workflow.name}")
    print("Steps:", list(workflow._steps.keys()))

    # Generate Mermaid diagram
    print("\nWorkflow Diagram (Mermaid):")
    print(workflow.visualize())

    # ============================================================
    # Example 4: Router for Specialized Agents
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 4: Router - Intelligent Routing")
    print("=" * 60)

    # Create specialized agents
    code_agent = Agent(
        "coder",
        model="gpt-4o-mini",
        system_prompt="You are a coding expert. Help with programming questions.",
    )

    math_agent = Agent(
        "mathematician",
        model="gpt-4o-mini",
        system_prompt="You are a math expert. Solve mathematical problems.",
    )

    general_agent = Agent(
        "assistant",
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant for general questions.",
    )

    # Create router with keyword-based routing
    router = Router(
        name="smart_router",
        strategy=RouteStrategy.KEYWORD,
        default_agent=general_agent,
    )

    router.add_route(
        name="code",
        agent=code_agent,
        keywords=["code", "programming", "python", "javascript", "function", "debug"],
        priority=10,
    )

    router.add_route(
        name="math",
        agent=math_agent,
        keywords=["calculate", "math", "equation", "solve", "number", "algebra"],
        priority=5,
    )

    print(f"\nRouter created: {router}")
    print("Routes:", list(router._routes.keys()))

    # Test routing decisions
    test_inputs = [
        "Write a Python function to sort a list",
        "Calculate the square root of 144",
        "What's the weather like today?",
    ]

    print("\nRouting examples:")
    for inp in test_inputs:
        routes = router._determine_routes(inp)
        routed_to = [r.name for r in routes] if routes else ["default"]
        print(f"  '{inp[:40]}...' → {routed_to}")

    # ============================================================
    # Example 5: Hierarchical Team with Leader
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 5: Hierarchical Team")
    print("=" * 60)

    from agentkit.orchestration.team import TeamRole

    leader = Agent(
        "leader",
        model="gpt-4o-mini",
        system_prompt="You are a team leader. Distribute tasks to your workers.",
    )

    worker1 = Agent("worker_1", model="gpt-4o-mini")
    worker2 = Agent("worker_2", model="gpt-4o-mini")

    hierarchical_team = Team("hierarchical", strategy=TeamStrategy.HIERARCHICAL)
    hierarchical_team.add_agent(leader, role=TeamRole.LEADER)
    hierarchical_team.add_agent(worker1, role=TeamRole.WORKER)
    hierarchical_team.add_agent(worker2, role=TeamRole.WORKER)

    print(f"\nHierarchical team: {hierarchical_team}")
    print(f"Leader: {hierarchical_team._leader.name if hierarchical_team._leader else 'None'}")
    print(f"Workers: {[a.name for a in hierarchical_team.get_agents(TeamRole.WORKER)]}")


if __name__ == "__main__":
    asyncio.run(main())
