"""LangGraph subgraph implementations for email processing and AI classification.

This module contains two main subgraphs:
1. Email Processing Subgraph - Deterministic filter pipeline
2. AI Classification Subgraph - AI-based email classification

Both subgraphs are designed to work with LangGraph's Send() API for parallel
processing while maintaining thread-safe state updates via reducers.
"""

import asyncio

import ollama

from .agents import (
    check_attachments,
    check_keywords,
    check_sender_pattern,
    check_whitelist,
    truncate_symmetric,
)
from .langgraph_state import GraphState, email_dict_to_model, sender_stats_dict_to_model
from .state import (
    Config,
    Decision,
    Email,
    EmailDecision,
    FilterReason,
    SenderStats,
)

# ============================================================================
# Email Processing Subgraph - Deterministic Filters
# ============================================================================


def check_attachments_node(state: GraphState) -> GraphState:
    """Node: Check if email has important attachments.

    Ported from attachment_filter_agent in agents.py.
    Makes immediate KEEP decision if important attachments found.

    Args:
        state: GraphState with current_email_uid and current_email

    Returns:
        Updated GraphState with decision (if made) or unchanged state
    """
    current_email_dict = state.get("current_email")
    if not current_email_dict:
        return state

    # Convert dict to Email model
    email = email_dict_to_model(current_email_dict)

    # Get config
    config = Config(**state["config"])

    # Run attachment check
    decision = check_attachments(email, config)

    if decision:
        # Create decision dict for state
        decision_dict = {
            "email": email.model_dump(),
            "decision": decision.decision.value,
            "reason": decision.reason.value,
            "confidence": decision.confidence,
            "processed_at": decision.processed_at.isoformat(),
        }

        # Update sender stats
        sender_stats_update = _create_sender_stats_update(
            email.sender, decision.decision, config
        )

        return {
            **state,
            "decisions": [decision_dict],
            "sender_stats": sender_stats_update,
            "processed_uids": {email.uid},
            "total_processed": state.get("total_processed", 0) + 1,
            "total_kept": state.get("total_kept", 0) + 1,
        }

    # No decision made, continue to next filter
    return state


def check_keywords_node(state: GraphState) -> GraphState:
    """Node: Check if email contains critical keywords.

    Ported from keyword_filter_agent in agents.py.
    Only runs if no prior decision was made.

    Args:
        state: GraphState with current_email_uid and current_email

    Returns:
        Updated GraphState with decision (if made) or unchanged state
    """
    # Check if decision already made by previous node
    if state.get("decisions"):
        return state

    current_email_dict = state.get("current_email")
    if not current_email_dict:
        return state

    # Convert dict to Email model
    email = email_dict_to_model(current_email_dict)

    # Get config
    config = Config(**state["config"])

    # Run keyword check
    decision = check_keywords(email, config)

    if decision:
        # Create decision dict for state
        decision_dict = {
            "email": email.model_dump(),
            "decision": decision.decision.value,
            "reason": decision.reason.value,
            "confidence": decision.confidence,
            "processed_at": decision.processed_at.isoformat(),
        }

        # Update sender stats
        sender_stats_update = _create_sender_stats_update(
            email.sender, decision.decision, config
        )

        return {
            **state,
            "decisions": [decision_dict],
            "sender_stats": sender_stats_update,
            "processed_uids": {email.uid},
            "total_processed": state.get("total_processed", 0) + 1,
            "total_kept": state.get("total_kept", 0) + 1,
        }

    # No decision made, continue to next filter
    return state


def check_whitelist_node(state: GraphState) -> GraphState:
    """Node: Check if email is from whitelisted domain.

    Ported from whitelist_filter_agent in agents.py.
    Only runs if no prior decision was made.

    Args:
        state: GraphState with current_email_uid and current_email

    Returns:
        Updated GraphState with decision (if made) or unchanged state
    """
    # Check if decision already made by previous node
    if state.get("decisions"):
        return state

    current_email_dict = state.get("current_email")
    if not current_email_dict:
        return state

    # Convert dict to Email model
    email = email_dict_to_model(current_email_dict)

    # Get config
    config = Config(**state["config"])

    # Run whitelist check
    decision = check_whitelist(email, config)

    if decision:
        # Create decision dict for state
        decision_dict = {
            "email": email.model_dump(),
            "decision": decision.decision.value,
            "reason": decision.reason.value,
            "confidence": decision.confidence,
            "processed_at": decision.processed_at.isoformat(),
        }

        # Update sender stats
        sender_stats_update = _create_sender_stats_update(
            email.sender, decision.decision, config
        )

        return {
            **state,
            "decisions": [decision_dict],
            "sender_stats": sender_stats_update,
            "processed_uids": {email.uid},
            "total_processed": state.get("total_processed", 0) + 1,
            "total_kept": state.get("total_kept", 0) + 1,
        }

    # No decision made, continue to next filter
    return state


def check_sender_pattern_node(state: GraphState) -> GraphState:
    """Node: Check if sender should be auto-deleted based on history.

    Ported from sender_pattern_filter_agent in agents.py.
    Only runs if no prior decision was made and sender tracking is enabled.

    Args:
        state: GraphState with current_email_uid and current_email

    Returns:
        Updated GraphState with decision (if made) or unchanged state
    """
    # Check if decision already made by previous node
    if state.get("decisions"):
        return state

    current_email_dict = state.get("current_email")
    if not current_email_dict:
        return state

    # Convert dict to Email model
    email = email_dict_to_model(current_email_dict)

    # Get config
    config = Config(**state["config"])

    # Check if sender tracking is enabled
    if not config.enable_sender_tracking:
        return state

    # Get sender stats from state
    sender_stats_dict = state.get("sender_stats", {}).get(email.sender)
    if sender_stats_dict:
        sender_stats = sender_stats_dict_to_model(sender_stats_dict)
    else:
        sender_stats = SenderStats(sender=email.sender)

    # Run sender pattern check
    decision = check_sender_pattern(email, sender_stats, config)

    if decision:
        # Create decision dict for state
        decision_dict = {
            "email": email.model_dump(),
            "decision": decision.decision.value,
            "reason": decision.reason.value,
            "confidence": decision.confidence,
            "processed_at": decision.processed_at.isoformat(),
        }

        # Update sender stats
        sender_stats_update = _create_sender_stats_update(
            email.sender, decision.decision, config
        )

        return {
            **state,
            "decisions": [decision_dict],
            "sender_stats": sender_stats_update,
            "processed_uids": {email.uid},
            "total_processed": state.get("total_processed", 0) + 1,
            "total_deleted": state.get("total_deleted", 0) + 1,
            "to_delete": [email.uid],
        }

    # No decision made, continue to final decision node
    return state


def final_decision_node(state: GraphState) -> GraphState:
    """Node: Make final decision for emails that passed all filters.

    If all deterministic filters passed without making a decision,
    mark the email as needing AI classification.

    Args:
        state: GraphState with current_email_uid and current_email

    Returns:
        Updated GraphState with email marked for full fetch
    """
    # Check if decision already made by previous nodes
    if state.get("decisions"):
        return state

    current_email_uid = state.get("current_email_uid")
    current_email_dict = state.get("current_email")

    if not current_email_uid or not current_email_dict:
        return state

    # Convert dict to Email model
    email = email_dict_to_model(current_email_dict)

    # Get config
    Config(**state["config"])

    # No decision made - mark for AI classification
    # Update sender stats (increment total count only, no marketing count yet)
    sender_stats_dict = state.get("sender_stats", {}).get(email.sender)
    if sender_stats_dict:
        current_stats = sender_stats_dict_to_model(sender_stats_dict)
    else:
        current_stats = SenderStats(sender=email.sender)

    sender_stats_update = {
        email.sender: {
            "sender": email.sender,
            "marketing_count": current_stats.marketing_count,
            "total_count": current_stats.total_count + 1,
            "auto_delete": current_stats.auto_delete,
        }
    }

    return {
        **state,
        "needs_full_fetch": [current_email_uid],
        "sender_stats": sender_stats_update,
        "processed_uids": {email.uid},
    }


# ============================================================================
# AI Classification Subgraph
# ============================================================================


async def classify_with_ai_async(email: Email, config: Config) -> EmailDecision:
    """Classify email using Ollama LLM with async support.

    Async version of classify_with_ai from agents.py.
    Uses Ollama AsyncClient for parallel processing.

    Args:
        email: Email to classify
        config: Pipeline configuration

    Returns:
        EmailDecision with AI classification result
    """
    # Strip HTML and truncate symmetrically to preserve footer
    body_preview = truncate_symmetric(email.body)

    prompt = f"""Classify this email as marketing/promotional or important.

Subject: {email.subject}
From: {email.sender}
Body preview: {body_preview}

Is this a marketing/promotional email that can be safely deleted?
Answer only YES or NO.

Answer:"""

    try:
        # Use async Ollama client
        client = ollama.AsyncClient(host=config.ollama_base_url)

        response = await client.chat(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response["message"]["content"].strip().upper()

        if "YES" in answer:
            decision = Decision.DELETE
        else:
            decision = Decision.KEEP

        return EmailDecision(
            email=email,
            decision=decision,
            reason=FilterReason.AI_CLASSIFIED,
            confidence=0.8,
        )

    except Exception as e:
        # On error, default to UNCERTAIN with low confidence
        print(f"[AI Classification Error] UID {email.uid}: {e}")
        return EmailDecision(
            email=email,
            decision=Decision.UNCERTAIN,
            reason=FilterReason.AI_CLASSIFIED,
            confidence=0.0,
        )


def ai_classification_node(state: GraphState) -> GraphState:
    """Node: Classify email using AI (Ollama).

    Ported from ai_classifier_agent in agents.py with async support.
    Runs AI classification for emails that passed deterministic filters.

    Args:
        state: GraphState with current_email (full body)

    Returns:
        Updated GraphState with AI decision
    """
    current_email_dict = state.get("current_email")
    if not current_email_dict:
        return state

    # Convert dict to Email model
    email = email_dict_to_model(current_email_dict)

    # Get config
    config = Config(**state["config"])

    # Run AI classification (sync wrapper for async function)
    decision = asyncio.run(classify_with_ai_async(email, config))

    # Create decision dict for state
    decision_dict = {
        "email": email.model_dump(),
        "decision": decision.decision.value,
        "reason": decision.reason.value,
        "confidence": decision.confidence,
        "processed_at": decision.processed_at.isoformat(),
    }

    # Update sender stats based on AI decision
    sender_stats_update = _create_sender_stats_update(
        email.sender, decision.decision, config
    )

    # Build state update
    state_update = {
        **state,
        "decisions": [decision_dict],
        "sender_stats": sender_stats_update,
        "total_processed": state.get("total_processed", 0) + 1,
    }

    # Add to appropriate counters and lists
    if decision.decision == Decision.DELETE:
        state_update["total_deleted"] = state.get("total_deleted", 0) + 1
        state_update["to_delete"] = [email.uid]
    elif decision.decision == Decision.KEEP:
        state_update["total_kept"] = state.get("total_kept", 0) + 1

    return state_update


def ai_final_decision_node(state: GraphState) -> GraphState:
    """Node: Finalize AI classification decision.

    This node ensures the decision is properly recorded in the state.
    It's a simple pass-through that validates the AI decision was made.

    Args:
        state: GraphState with AI decision

    Returns:
        Unchanged GraphState (decision already recorded)
    """
    # Decision was already made and recorded in ai_classification_node
    # This node is just a placeholder for potential future logic
    return state


# ============================================================================
# Helper Functions
# ============================================================================


def _create_sender_stats_update(
    sender: str, decision: Decision, config: Config
) -> dict[str, dict]:
    """Create sender stats update dict for state merge.

    Helper function to create properly formatted sender stats update
    that will be merged via the merge_sender_stats reducer.

    Args:
        sender: Email sender address
        decision: Decision made for this email
        config: Pipeline configuration

    Returns:
        Dict mapping sender to stats dict (for reducer)
    """
    # Increment marketing count if DELETE decision
    marketing_increment = 1 if decision == Decision.DELETE else 0

    # Check if should enable auto_delete
    # Note: This is a delta update, the reducer will handle merging
    auto_delete = marketing_increment >= config.auto_delete_threshold

    return {
        sender: {
            "sender": sender,
            "marketing_count": marketing_increment,
            "total_count": 1,
            "auto_delete": auto_delete,
        }
    }


# ============================================================================
# Subgraph Assembly Functions
# ============================================================================


def create_email_processing_subgraph_nodes() -> dict[str, callable]:
    """Get node mapping for email processing subgraph.

    Returns:
        Dict mapping node names to node functions
    """
    return {
        "check_attachments": check_attachments_node,
        "check_keywords": check_keywords_node,
        "check_whitelist": check_whitelist_node,
        "check_sender_pattern": check_sender_pattern_node,
        "final_decision": final_decision_node,
    }


def create_ai_classification_subgraph_nodes() -> dict[str, callable]:
    """Get node mapping for AI classification subgraph.

    Returns:
        Dict mapping node names to node functions
    """
    return {
        "ai_classification": ai_classification_node,
        "ai_final_decision": ai_final_decision_node,
    }
