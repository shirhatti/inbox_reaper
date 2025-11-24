"""LangGraph state schema with TypedDict and reducers for parallel processing.

This module defines the state schema for LangGraph-based orchestration of the
email classification pipeline. It uses TypedDict for state shape and Annotated
reducers for thread-safe synchronization across parallel subgraphs.
"""

from typing import Annotated, TypedDict
import operator

from .state import EmailDecision, SenderStats, Config, Email


def merge_sender_stats(
    existing: dict[str, dict], updates: dict[str, dict]
) -> dict[str, dict]:
    """Thread-safe merge of sender statistics from parallel pipelines.

    This reducer handles race conditions when multiple parallel email processing
    subgraphs update sender statistics simultaneously.

    Args:
        existing: Current sender stats in state
        updates: New sender stats from parallel subgraph

    Returns:
        Merged sender stats with accumulated counts
    """
    result = existing.copy()

    for sender, new_stats_dict in updates.items():
        if sender in result:
            # Merge counts from parallel updates
            existing_stats = result[sender]
            result[sender] = {
                "sender": sender,
                "marketing_count": existing_stats["marketing_count"]
                + new_stats_dict["marketing_count"],
                "total_count": existing_stats["total_count"]
                + new_stats_dict["total_count"],
                "auto_delete": new_stats_dict["auto_delete"]
                or existing_stats["auto_delete"],
            }
        else:
            result[sender] = new_stats_dict

    return result


def merge_email_headers(
    existing: dict[str, dict], updates: dict[str, dict]
) -> dict[str, dict]:
    """Merge email headers by UID.

    Args:
        existing: Current email headers in state
        updates: New email headers to merge

    Returns:
        Merged email headers dictionary
    """
    result = existing.copy()
    result.update(updates)
    return result


class GraphState(TypedDict):
    """LangGraph state schema for email classification pipeline.

    This TypedDict defines the shape of state that flows through the LangGraph.
    Annotated fields use reducers for thread-safe parallel updates.

    State Flow:
        1. batch_fetch_headers → populates email_headers
        2. fan_out_processing → spawns parallel subgraphs per email
        3. Parallel subgraphs update: decisions, sender_stats, needs_full_fetch
        4. aggregate_results → collects all results
        5. [conditional] batch_fetch_bodies → fetches full messages if needed
        6. [conditional] fan_out_ai_classification → parallel AI processing
        7. batch_delete → executes IMAP delete for marked emails
        8. update_checkpoint → persists progress
    """

    # Configuration (set once at start)
    config: dict  # Serialized Config object

    # ===== Batch Data (populated by fetch nodes) =====
    # Email headers fetched from IMAP (UID -> header dict)
    email_headers: Annotated[dict[str, dict], merge_email_headers]

    # Full email bodies fetched for AI classification (UID -> Email dict)
    email_bodies: Annotated[dict[str, dict], merge_email_headers]

    # ===== Shared Coordination State (updated by parallel subgraphs) =====
    # Sender statistics - synchronized across parallel pipelines
    # Using custom reducer to handle concurrent updates
    sender_stats: Annotated[dict[str, dict], merge_sender_stats]

    # All decisions made (append-only from parallel subgraphs)
    decisions: Annotated[list[dict], operator.add]

    # UIDs that need full body fetch for AI classification
    needs_full_fetch: Annotated[list[str], operator.add]

    # UIDs marked for deletion
    to_delete: Annotated[list[str], operator.add]

    # ===== Progress Tracking =====
    # All processed UIDs (union from all subgraphs)
    processed_uids: Annotated[set[str], operator.or_]

    # Errors encountered (append from any subgraph)
    errors: Annotated[list[dict], operator.add]

    # Counters
    total_processed: int
    total_deleted: int
    total_kept: int

    # ===== Pagination State =====
    # Track pagination for resumable processing
    min_uid: str | None
    max_uid: str | None
    consecutive_empty_batches: int

    # ===== IMAP Connection Info =====
    # IMAP session details for batch operations
    imap_server: str | None
    imap_username: str | None


def create_initial_state(config: Config) -> GraphState:
    """Create initial LangGraph state from configuration.

    Args:
        config: Pipeline configuration

    Returns:
        Initial GraphState ready for processing
    """
    return GraphState(
        config=config.model_dump(),
        email_headers={},
        email_bodies={},
        sender_stats={},
        decisions=[],
        needs_full_fetch=[],
        to_delete=[],
        processed_uids=set(),
        errors=[],
        total_processed=0,
        total_deleted=0,
        total_kept=0,
        min_uid=None,
        max_uid=None,
        consecutive_empty_batches=0,
        imap_server=None,
        imap_username=None,
    )


def email_dict_to_model(email_dict: dict) -> Email:
    """Convert email dictionary from state to Email model.

    Args:
        email_dict: Serialized email from state

    Returns:
        Email Pydantic model
    """
    return Email(**email_dict)


def decision_dict_to_model(decision_dict: dict) -> EmailDecision:
    """Convert decision dictionary from state to EmailDecision model.

    Args:
        decision_dict: Serialized decision from state

    Returns:
        EmailDecision Pydantic model
    """
    return EmailDecision(
        email=Email(**decision_dict["email"]),
        decision=decision_dict["decision"],
        reason=decision_dict["reason"],
        confidence=decision_dict["confidence"],
        processed_at=decision_dict["processed_at"],
    )


def sender_stats_dict_to_model(stats_dict: dict) -> SenderStats:
    """Convert sender stats dictionary to SenderStats model.

    Args:
        stats_dict: Serialized sender stats from state

    Returns:
        SenderStats Pydantic model
    """
    return SenderStats(**stats_dict)
