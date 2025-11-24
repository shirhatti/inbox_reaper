"""LangGraph DAG definition for parallel email classification workflow.

This module implements the main LangGraph orchestration for email processing
with parallel pipelines, synchronization, and batch operations.

Graph Structure:
    START → batch_fetch_headers → fan_out_processing → aggregate_results
         → [conditional: needs_bodies?] → batch_fetch_bodies → fan_out_ai
         → aggregate_ai_results → batch_delete → update_checkpoint → END
"""

from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .langgraph_state import GraphState, create_initial_state
from .state import Config

# ===== Main Graph Nodes =====


def batch_fetch_headers(state: GraphState) -> GraphState:
    """Fetch email headers from IMAP in batch.

    This node connects to IMAP and fetches headers for a batch of emails.
    Headers include: UID, subject, sender, date, attachment info (but not body).

    Args:
        state: Current graph state

    Returns:
        Updated state with email_headers populated
    """
    # TODO: Implement IMAP connection and header fetch
    # For now, this is a placeholder that will be implemented with IMAP integration

    config = Config(**state["config"])
    fetch_size = config.fetch_size

    print(f"[batch_fetch_headers] Fetching up to {fetch_size} email headers...")

    # Placeholder: In production, this will:
    # 1. Connect to IMAP server
    # 2. Search for unprocessed emails
    # 3. Fetch headers only (BODY.PEEK[HEADER])
    # 4. Parse headers into email_headers dict

    email_headers = {}  # UID -> {uid, subject, sender, date, attachments}

    # Example structure:
    # email_headers = {
    #     "12345": {
    #         "uid": "12345",
    #         "subject": "Newsletter from Company",
    #         "sender": "news@company.com",
    #         "date": "2025-11-24T10:30:00",
    #         "attachments": [],
    #         "body": "",  # Empty for header-only fetch
    #     }
    # }

    print(f"[batch_fetch_headers] Fetched {len(email_headers)} email headers")

    return {
        **state,
        "email_headers": email_headers,
    }


def fan_out_processing(state: GraphState) -> list[Send]:
    """Fan out to parallel email processing subgraphs.

    This node spawns a separate subgraph for each email to process them
    in parallel. Each subgraph runs the deterministic filter pipeline.

    Args:
        state: Current graph state with email_headers

    Returns:
        List of Send commands to spawn parallel subgraphs
    """
    email_headers = state["email_headers"]

    print(f"[fan_out_processing] Spawning {len(email_headers)} parallel subgraphs...")

    # Spawn parallel subgraph for each email
    # Each subgraph will run: attachment → keyword → whitelist → sender_pattern
    sends = []
    for uid, email_header in email_headers.items():
        sends.append(
            Send(
                "email_processing_subgraph",
                {
                    **state,
                    "current_email_uid": uid,
                    "current_email": email_header,
                },
            )
        )

    return sends


def aggregate_results(state: GraphState) -> GraphState:
    """Aggregate results from all parallel email processing subgraphs.

    This node collects all decisions, sender stats updates, and emails
    that need full body fetch for AI classification.

    Args:
        state: Current graph state with accumulated decisions

    Returns:
        Updated state with aggregated results
    """
    decisions = state["decisions"]
    needs_full_fetch = state["needs_full_fetch"]
    sender_stats = state["sender_stats"]

    print(f"[aggregate_results] Aggregated {len(decisions)} decisions")
    print(f"[aggregate_results] {len(needs_full_fetch)} emails need AI classification")
    print(f"[aggregate_results] Tracking {len(sender_stats)} senders")

    # State is already aggregated by reducers, just log summary
    return state


def should_fetch_bodies(state: GraphState) -> Literal["fetch_bodies", "skip_bodies"]:
    """Conditional edge: Check if any emails need full body fetch for AI.

    Args:
        state: Current graph state

    Returns:
        "fetch_bodies" if any emails need AI classification, else "skip_bodies"
    """
    needs_full_fetch = state["needs_full_fetch"]

    if needs_full_fetch:
        print(
            f"[should_fetch_bodies] Need to fetch {len(needs_full_fetch)} email bodies"
        )
        return "fetch_bodies"
    else:
        print("[should_fetch_bodies] No emails need AI classification, skipping")
        return "skip_bodies"


def batch_fetch_bodies(state: GraphState) -> GraphState:
    """Fetch full email bodies for AI classification in batch.

    This node fetches complete email bodies only for emails that need
    AI classification (to minimize IMAP bandwidth).

    Args:
        state: Current graph state with needs_full_fetch list

    Returns:
        Updated state with email_bodies populated
    """
    needs_full_fetch = state["needs_full_fetch"]

    print(f"[batch_fetch_bodies] Fetching {len(needs_full_fetch)} full email bodies...")

    # TODO: Implement batch IMAP body fetch
    # For now, this is a placeholder

    # Placeholder: In production, this will:
    # 1. Connect to IMAP server (reuse connection if possible)
    # 2. Fetch full bodies for UIDs in needs_full_fetch
    # 3. Use IMAP pipelining for efficiency (BODY.PEEK[TEXT])
    # 4. Parse and store in email_bodies

    email_bodies = {}  # UID -> complete Email dict with body

    print(f"[batch_fetch_bodies] Fetched {len(email_bodies)} email bodies")

    return {
        **state,
        "email_bodies": email_bodies,
    }


def fan_out_ai_classification(state: GraphState) -> list[Send]:
    """Fan out to parallel AI classification subgraphs.

    This node spawns parallel subgraphs for AI classification of emails
    that need full analysis.

    Args:
        state: Current graph state with email_bodies

    Returns:
        List of Send commands for parallel AI classification
    """
    email_bodies = state["email_bodies"]

    print(
        f"[fan_out_ai_classification] Spawning {len(email_bodies)} "
        "AI classification tasks..."
    )

    # Spawn parallel AI classification subgraph for each email
    sends = []
    for uid, email_body in email_bodies.items():
        sends.append(
            Send(
                "ai_classification_subgraph",
                {
                    **state,
                    "current_email_uid": uid,
                    "current_email": email_body,
                },
            )
        )

    return sends


def aggregate_ai_results(state: GraphState) -> GraphState:
    """Aggregate results from AI classification subgraphs.

    Args:
        state: Current graph state with AI decisions

    Returns:
        Updated state with all final decisions
    """
    decisions = state["decisions"]

    print(f"[aggregate_ai_results] Total decisions after AI: {len(decisions)}")

    return state


def batch_delete(state: GraphState) -> GraphState:
    """Execute batch delete operation on IMAP for all marked emails.

    This node performs a single batch IMAP delete operation for all
    emails marked for deletion, maximizing efficiency.

    Args:
        state: Current graph state with to_delete list

    Returns:
        Updated state with deletion results
    """
    to_delete = state["to_delete"]
    config = Config(**state["config"])

    print(f"[batch_delete] Marking {len(to_delete)} emails for deletion...")

    if config.dry_run:
        print("[batch_delete] DRY RUN - No actual deletions performed")
    else:
        # TODO: Implement batch IMAP delete
        # For now, this is a placeholder

        # Placeholder: In production, this will:
        # 1. Connect to IMAP server
        # 2. Mark all UIDs in to_delete with \Deleted flag
        # 3. Execute EXPUNGE to permanently delete
        # 4. Handle errors gracefully

        print(f"[batch_delete] Would delete UIDs: {to_delete[:10]}...")

    return {
        **state,
        "total_deleted": state["total_deleted"] + len(to_delete),
    }


def update_checkpoint(state: GraphState) -> GraphState:
    """Update checkpoint with processing progress.

    This node persists state to SQLite for resumability.

    Args:
        state: Current graph state

    Returns:
        Updated state with checkpoint saved
    """
    print("[update_checkpoint] Saving checkpoint...")

    # Checkpoint is automatically handled by SqliteSaver
    # This node just logs progress

    print(
        f"[update_checkpoint] Processed: {state['total_processed']}, "
        f"Deleted: {state['total_deleted']}, "
        f"Kept: {state['total_kept']}"
    )

    return state


# ===== Subgraph Nodes (will be implemented separately) =====


def email_processing_subgraph(state: GraphState) -> GraphState:
    """Process a single email through deterministic filter pipeline.

    This subgraph runs for each email in parallel:
        1. Check attachments
        2. Check keywords
        3. Check whitelist
        4. Check sender pattern
        5. Make decision or mark for AI classification

    Args:
        state: Graph state with current_email and current_email_uid

    Returns:
        Updated state with decision or needs_full_fetch marked
    """
    # TODO: Implement in separate file (langgraph_subgraphs.py)
    # For now, placeholder that marks all emails as needing AI

    current_email_uid = state.get("current_email_uid")
    print(f"[email_processing_subgraph] Processing UID {current_email_uid}")

    # Placeholder: In production, this will run full deterministic pipeline
    # For now, just mark as needing AI classification

    return {
        **state,
        "needs_full_fetch": [current_email_uid],
    }


def ai_classification_subgraph(state: GraphState) -> GraphState:
    """Classify a single email using AI (Ollama).

    This subgraph runs AI classification for emails that passed through
    deterministic filters without a decision.

    Args:
        state: Graph state with current_email (full body)

    Returns:
        Updated state with AI decision
    """
    # TODO: Implement in separate file (langgraph_subgraphs.py)
    # For now, placeholder

    current_email_uid = state.get("current_email_uid")
    print(f"[ai_classification_subgraph] AI classifying UID {current_email_uid}")

    # Placeholder: In production, this will call Ollama for classification

    return state


# ===== Build LangGraph =====


def build_langgraph(
    config: Config, checkpoint_path: str = "checkpoints.db"
) -> StateGraph:
    """Build the main LangGraph for email classification pipeline.

    Args:
        config: Pipeline configuration
        checkpoint_path: Path to SQLite checkpoint database

    Returns:
        Compiled LangGraph ready for execution
    """
    # Initialize checkpointer for state persistence
    checkpointer = SqliteSaver.from_conn_string(checkpoint_path)

    # Create the graph
    graph = StateGraph(GraphState)

    # Add main workflow nodes
    graph.add_node("batch_fetch_headers", batch_fetch_headers)
    graph.add_node("fan_out_processing", fan_out_processing)
    graph.add_node("aggregate_results", aggregate_results)
    graph.add_node("batch_fetch_bodies", batch_fetch_bodies)
    graph.add_node("fan_out_ai_classification", fan_out_ai_classification)
    graph.add_node("aggregate_ai_results", aggregate_ai_results)
    graph.add_node("batch_delete", batch_delete)
    graph.add_node("update_checkpoint", update_checkpoint)

    # Add subgraph nodes (these will be called by Send())
    graph.add_node("email_processing_subgraph", email_processing_subgraph)
    graph.add_node("ai_classification_subgraph", ai_classification_subgraph)

    # Define main workflow edges
    graph.add_edge(START, "batch_fetch_headers")
    graph.add_edge("batch_fetch_headers", "fan_out_processing")
    graph.add_edge("fan_out_processing", "aggregate_results")

    # Conditional edge: fetch bodies only if needed for AI
    graph.add_conditional_edges(
        "aggregate_results",
        should_fetch_bodies,
        {
            "fetch_bodies": "batch_fetch_bodies",
            "skip_bodies": "batch_delete",
        },
    )

    graph.add_edge("batch_fetch_bodies", "fan_out_ai_classification")
    graph.add_edge("fan_out_ai_classification", "aggregate_ai_results")
    graph.add_edge("aggregate_ai_results", "batch_delete")
    graph.add_edge("batch_delete", "update_checkpoint")
    graph.add_edge("update_checkpoint", END)

    # Compile graph with checkpointer
    compiled_graph = graph.compile(checkpointer=checkpointer)

    print("[build_langgraph] LangGraph compiled successfully")
    return compiled_graph


def run_langgraph_pipeline(config: Config, checkpoint_path: str = "checkpoints.db"):
    """Run the complete LangGraph email classification pipeline.

    Args:
        config: Pipeline configuration
        checkpoint_path: Path to SQLite checkpoint database
    """
    # Build graph
    graph = build_langgraph(config, checkpoint_path)

    # Create initial state
    initial_state = create_initial_state(config)

    # Run graph
    print("[run_langgraph_pipeline] Starting pipeline execution...")

    # Execute graph with automatic checkpointing
    final_state = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "email_classification_session"}},
    )

    print("[run_langgraph_pipeline] Pipeline execution complete")
    print(f"Total processed: {final_state['total_processed']}")
    print(f"Total deleted: {final_state['total_deleted']}")
    print(f"Total kept: {final_state['total_kept']}")

    return final_state
