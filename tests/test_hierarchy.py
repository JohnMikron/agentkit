from unittest.mock import AsyncMock

import pytest

from agentkit.core.agent import Agent
from agentkit.core.types import AgentResult
from agentkit.orchestration.hierarchy import DelegatedTask, HierarchicalTeam, SupervisorResponse


@pytest.fixture
def mock_supervisor():
    supervisor = Agent("Supervisor")
    supervisor.arun_structured = AsyncMock()
    return supervisor

@pytest.fixture
def mock_workers():
    from agentkit.core.types import AgentResult
    researcher = Agent("Researcher", system_prompt="I research things")
    researcher.arun = AsyncMock(return_value=AgentResult(content="Research complete.", success=True))

    writer = Agent("Writer", system_prompt="I write things")
    writer.arun = AsyncMock(return_value=AgentResult(content="Writing complete.", success=True))

    return [researcher, writer]


@pytest.mark.asyncio
async def test_hierarchical_team_success(mock_supervisor, mock_workers):
    # Setup team
    team = HierarchicalTeam(supervisor=mock_supervisor, workers=mock_workers, max_iterations=3)

    # Mock supervisor returning delegations then finishing
    mock_supervisor.arun_structured.side_effect = [
        AgentResult(data=SupervisorResponse(
            thoughts="I need to research first",
            delegations=[DelegatedTask(worker_name="Researcher", instructions="Find data")],
            is_finished=False,
            final_answer=""
        )),
        AgentResult(data=SupervisorResponse(
            thoughts="Now I need to write",
            delegations=[DelegatedTask(worker_name="Writer", instructions="Write draft")],
            is_finished=False,
            final_answer=""
        )),
        AgentResult(data=SupervisorResponse(
            thoughts="All done",
            delegations=[],
            is_finished=True,
            final_answer="Final Result"
        ))
    ]

    # Run team
    result = await team.arun("Write a report")

    assert result == "Final Result"
    assert mock_supervisor.arun_structured.call_count == 3
    assert mock_workers[0].arun.call_count == 1  # Researcher
    assert mock_workers[1].arun.call_count == 1  # Writer

    # Check that worker was called with correct instructions
    mock_workers[0].arun.assert_called_with("Find data")
    mock_workers[1].arun.assert_called_with("Write draft")

@pytest.mark.asyncio
async def test_hierarchical_team_max_iterations(mock_supervisor, mock_workers):
    team = HierarchicalTeam(supervisor=mock_supervisor, workers=mock_workers, max_iterations=2)

    # Always return delegations without finishing
    mock_supervisor.arun_structured.return_value = AgentResult(data=SupervisorResponse(
        thoughts="Still working",
        delegations=[DelegatedTask(worker_name="Researcher", instructions="More data")],
        is_finished=False,
        final_answer=""
    ))

    with pytest.raises(Exception, match="HierarchicalTeam reached max iterations"):
        await team.arun("Impossible task")
