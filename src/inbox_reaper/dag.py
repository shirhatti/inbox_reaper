"""DAG definition for email classification workflow.

This module defines the agent workflow using Google's Agent Development Kit.
The workflow is a directed acyclic graph where each node is an agent function
that transforms the ProcessingState.
"""

from collections.abc import Callable

from google import genai
from google.genai import types

from .agents import (
    ai_classifier_agent,
    attachment_filter_agent,
    check_termination_agent,
    keyword_filter_agent,
    log_progress_agent,
    sender_pattern_filter_agent,
    whitelist_filter_agent,
)
from .state import ProcessingState

# Define the agent workflow as a sequence of transformations
# Each agent is a pure function: ProcessingState -> ProcessingState
AGENT_PIPELINE = [
    ("attachment_filter", attachment_filter_agent),
    ("keyword_filter", keyword_filter_agent),
    ("whitelist_filter", whitelist_filter_agent),
    ("sender_pattern_filter", sender_pattern_filter_agent),
    ("ai_classifier", ai_classifier_agent),
    ("log_progress", log_progress_agent),
    ("check_termination", check_termination_agent),
]


def create_agent_function(
    name: str, agent_fn: Callable[[ProcessingState], ProcessingState]
) -> types.FunctionDeclaration:
    """Create a FunctionDeclaration for use in Google ADK.

    This wraps our pure state transformation functions for use in the ADK.
    """
    return types.FunctionDeclaration(
        name=name,
        description=f"Execute {name} agent on current state",
        parameters={
            "type": "object",
            "properties": {
                "state": {
                    "type": "object",
                    "description": "Current processing state",
                }
            },
            "required": ["state"],
        },
    )


def build_dag(client: genai.Client) -> genai.Agent:
    """Build the agent DAG using Google's Agent Development Kit.

    The DAG is a linear pipeline for now, but can be extended to support
    parallel branches, conditional routing, etc.

    Args:
        client: Google GenAI client

    Returns:
        Configured agent that can process email batches
    """
    # Create function declarations for each agent
    agent_functions = [create_agent_function(name, fn) for name, fn in AGENT_PIPELINE]

    # Create the root agent
    # Note: For initial scaffold, we're using a simple sequential flow
    # In production, this would use the full ADK graph capabilities
    agent = client.agentic.create_agent(
        model="gemini-2.0-flash-exp",
        config=types.AgentConfig(
            name="email_classifier",
            description="Email classifier with hybrid deterministic + AI pipeline",
            instruction="""You are an email classification agent.

Process emails through a multi-layer decision pipeline:
1. Check for important attachments (deterministic)
2. Check for critical keywords (deterministic)
3. Check against whitelist (deterministic)
4. Check sender patterns (deterministic with history)
5. Classify with AI (expensive, only if needed)

Always process in this order to minimize AI calls.""",
            functions=agent_functions,
        ),
    )

    return agent


def run_pipeline(state: ProcessingState) -> ProcessingState:
    """Run the complete agent pipeline on the current state.

    This is a simple sequential executor for the agent pipeline.
    Each agent transforms the state and passes it to the next agent.

    Args:
        state: Initial processing state with emails to process

    Returns:
        Final processing state with all decisions made
    """
    current_state = state

    for name, agent_fn in AGENT_PIPELINE:
        try:
            print(f"\n→ Running {name}...")
            current_state = agent_fn(current_state)

            # Check for termination
            if current_state.errors:
                print(f"✗ Pipeline terminated: {current_state.errors[-1]}")
                break

        except Exception as e:
            error_msg = f"Error in {name}: {str(e)}"
            print(f"✗ {error_msg}")
            current_state = current_state.add_error(error_msg)
            break

    return current_state


def run_pipeline_with_adk(
    state: ProcessingState, client: genai.Client
) -> ProcessingState:
    """Run the pipeline using Google ADK agent.

    This is a placeholder for full ADK integration. For now, it falls back
    to the simple sequential pipeline.

    Args:
        state: Initial processing state
        client: Google GenAI client

    Returns:
        Final processing state
    """
    # TODO: Implement full ADK graph execution
    # For now, use simple sequential pipeline
    return run_pipeline(state)
