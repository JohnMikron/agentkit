"""
Example: Cyclic workflow routing using StateGraph.

This example demonstrates a customer support triage system that
evaluates a ticket, attempts to resolve it automatically, and routes
it to human escalation if necessary. It can retry gathering info in a loop.
"""

import asyncio
from agentkit import Agent
from agentkit.orchestration.stategraph import StateGraph

async def main():
    print("Initializing Agents...")
    
    # 1. Router Agent
    router = Agent(
        "TriageRouter", 
        system_prompt="You route customer tickets. Output exactly 'resolve_bot' for simple issues (like passwords), 'human_escalation' for complex issues, or 'need_info' if the user's message is too vague."
    )
    
    # 2. Solver Bot
    solver = Agent(
        "SolverBot",
        system_prompt="You provide immediate solutions to simple IT issues. Keep it brief."
    )
    
    # 3. Info Gatherer
    gatherer = Agent(
        "InfoGatherer",
        system_prompt="Ask the user for more details about their problem. Be polite."
    )
    
    print("Building StateGraph...")
    # We use a simple dictionary state for this example
    graph = StateGraph(dict)
    
    async def node_triage(state: dict) -> dict:
        ticket = state["ticket"]
        print(f"\n[Triage] Analyzing: '{ticket}'")
        res = await router.arun(ticket)
        decision = res.content.strip().lower()
        print(f"[Triage] Decision: {decision}")
        return {"route": decision}
        
    async def node_solve(state: dict) -> dict:
        print("[SolverBot] Processing...")
        res = await solver.arun(f"Solve this: {state['ticket']}")
        print(f"[SolverBot] Response: {res.content}")
        return {"status": "resolved"}
        
    async def node_gather(state: dict) -> dict:
        print("[GathererBot] Processing...")
        res = await gatherer.arun(state["ticket"])
        print(f"[GathererBot] Response: {res.content}")
        
        # In a real app, you'd pause and wait for user input.
        # Here we simulate the user providing more context.
        simulated_user_input = "Oh, my screen is completely blue and says error 404."
        print(f"[User Reply Simulated]: {simulated_user_input}")
        
        return {
            "ticket": state["ticket"] + " | Context: " + simulated_user_input,
            "gather_count": state.get("gather_count", 0) + 1
        }
        
    async def node_escalate(state: dict) -> dict:
        print("[System] Ticket escalated to human agents.")
        return {"status": "escalated"}
        
    # Add nodes
    graph.add_node("Triage", node_triage)
    graph.add_node("Solve", node_solve)
    graph.add_node("Gather", node_gather)
    graph.add_node("Escalate", node_escalate)
    
    # Set entry point
    graph.set_entry_point("Triage")
    
    # Conditional logic
    def routing_logic(state: dict) -> str:
        route = state.get("route", "")
        if "resolve" in route:
            return "Solve"
        elif "info" in route:
            if state.get("gather_count", 0) >= 2:
                print("[System] Too many info requests. Escalating.")
                return "Escalate"
            return "Gather"
        else:
            return "Escalate"
            
    # Edges
    graph.add_conditional_edge("Triage", routing_logic)
    
    # Loops back to Triage after gathering info
    graph.add_edge("Gather", "Triage")
    
    # Finish points
    graph.set_finish_point("Solve")
    graph.set_finish_point("Escalate")
    
    # Run example 1: Simple
    print("\n--- Test 1: Simple Password Issue ---")
    await graph.ainvoke({"ticket": "I forgot my password"})
    
    # Run example 2: Vague
    print("\n--- Test 2: Vague Issue (Cycles) ---")
    await graph.ainvoke({"ticket": "It is broken"})

if __name__ == "__main__":
    asyncio.run(main())
