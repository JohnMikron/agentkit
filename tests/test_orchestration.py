import pytest
import asyncio
from agentkit.core.agent import Agent
from agentkit.core.reflection import ReflectionAgent
from agentkit.orchestration.workflow import Workflow, TransitionType
from agentkit.providers.mock import MockProvider

@pytest.mark.asyncio
async def test_reflection_agent_basic():
    # Primary agent
    agent = Agent("writer")
    agent._provider = MockProvider(responses=["Initial draft", "Improved draft"])
    
    # Critique agent
    critic = Agent("critic")
    critic._provider = MockProvider(responses=["Critique 1"])
    
    reflector = ReflectionAgent(agent, critique_agent=critic, max_iterations=1)
    result = await reflector.arun("Write a story")
    
    assert result.content == "Improved draft"
    assert reflector.agent.name == "writer"
    assert reflector.critique_agent.name == "critic"

@pytest.mark.asyncio
async def test_workflow_basic():
    # Setup agents
    a1 = Agent("a1")
    a1._provider = MockProvider(responses=["Step 1 output"])
    a2 = Agent("a2")
    a2._provider = MockProvider(responses=["Step 2 output"])
    
    # Create workflow
    wf = Workflow("test_wf")
    wf.add_step("step1", a1, "Do step 1")
    wf.add_step("step2", a2, "Do step 2 with {{ step1_result }}")
    
    wf.add_transition("step1", "step2", TransitionType.ON_SUCCESS)
    
    result = await wf.arun("Start")
    
    assert result.success
    assert result.final_output == "Step 2 output"
    assert "step1" in result.state.completed_steps
    assert "step2" in result.state.completed_steps
    assert getattr(result.state.context, "step1_result") == "Step 1 output"

@pytest.mark.asyncio
async def test_workflow_conditional():
    a1 = Agent("a1")
    # First response will trigger success transition, second would fail
    a1._provider = MockProvider(responses=["Success"])
    
    a_success = Agent("success_agent")
    a_success._provider = MockProvider(responses=["Hooray!"])
    
    wf = Workflow("conditional_wf")
    wf.add_step("check", a1)
    wf.add_step("celebrate", a_success)
    
    # Conditional transition
    wf.add_transition("check", "celebrate", TransitionType.CONDITIONAL, 
                      condition=lambda r: "Success" in r.content)
    
    result = await wf.arun("Check status")
    assert "celebrate" in result.state.completed_steps
    assert result.final_output == "Hooray!"

def test_workflow_visualize():
    a1 = Agent("a1")
    wf = Workflow("viz")
    wf.add_step("s1", a1)
    wf.add_step("s2", a1)
    wf.add_transition("s1", "s2")
    
    mermaid = wf.visualize()
    assert "graph TD" in mermaid
    assert "s1" in mermaid
    assert "s2" in mermaid
