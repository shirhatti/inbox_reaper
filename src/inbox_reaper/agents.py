"""Agent functions for email classification.

All agents are pure functions: ProcessingState -> ProcessingState
Each agent processes emails in the current batch and adds decisions.
"""

import ollama

from .state import (
    Config,
    Decision,
    Email,
    EmailDecision,
    FilterReason,
    ProcessingState,
    SenderStats,
)


def check_attachments(email: Email, config: Config) -> EmailDecision | None:
    """Check if email has important attachments.

    Pure function that returns a KEEP decision if important attachments found.
    """
    for attachment in email.attachments:
        for ext in config.important_extensions:
            if attachment.lower().endswith(ext.lower()):
                return EmailDecision(
                    email=email,
                    decision=Decision.KEEP,
                    reason=FilterReason.ATTACHMENT,
                    confidence=1.0,
                )
    return None


def check_keywords(email: Email, config: Config) -> EmailDecision | None:
    """Check if email contains critical keywords.

    Pure function that returns a KEEP decision if keywords found.
    """
    text = f"{email.subject} {email.body}".lower()
    for keyword in config.keywords:
        if keyword.lower() in text:
            return EmailDecision(
                email=email,
                decision=Decision.KEEP,
                reason=FilterReason.KEYWORD,
                confidence=1.0,
            )
    return None


def check_whitelist(email: Email, config: Config) -> EmailDecision | None:
    """Check if email is from whitelisted domain.

    Pure function that returns a KEEP decision if sender is whitelisted.
    """
    # Extract domain from sender email
    if "@" in email.sender:
        domain = email.sender.split("@")[1].lower()
        for whitelisted in config.whitelist_domains:
            if domain == whitelisted.lower():
                return EmailDecision(
                    email=email,
                    decision=Decision.KEEP,
                    reason=FilterReason.WHITELIST,
                    confidence=1.0,
                )
    return None


def check_sender_pattern(
    email: Email, sender_stats: SenderStats, config: Config
) -> EmailDecision | None:
    """Check if sender should be auto-deleted based on history.

    Pure function that returns a DELETE decision if sender has pattern of marketing.
    """
    if sender_stats.should_auto_delete(config.auto_delete_threshold):
        return EmailDecision(
            email=email,
            decision=Decision.DELETE,
            reason=FilterReason.SENDER_PATTERN,
            confidence=1.0,
        )
    return None


def classify_with_ai(email: Email, config: Config) -> EmailDecision:
    """Classify email using Ollama LLM.

    This is the only non-pure function (has side effect of calling Ollama).
    Returns DELETE decision for marketing, KEEP for everything else.
    """
    # Truncate body to first 1000 chars for efficiency
    body_preview = email.body[:1000]

    prompt = f"""Classify this email as marketing/promotional or important.

Subject: {email.subject}
From: {email.sender}
Body preview: {body_preview}

Is this a marketing/promotional email that can be safely deleted?
Answer only YES or NO.

Answer:"""

    try:
        response = ollama.chat(
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

    except Exception:
        # On error, default to KEEP (safe default)
        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.AI_CLASSIFIED,
            confidence=0.0,
        )


# ============================================================================
# Agent Functions (Pure: ProcessingState -> ProcessingState)
# ============================================================================


def attachment_filter_agent(state: ProcessingState) -> ProcessingState:
    """Process emails through attachment filter.

    Pure function that checks each email for important attachments.
    """
    new_state = state

    for email in state.emails:
        # Skip if already processed
        if email.uid in state.processed_uids:
            continue

        decision = check_attachments(email, state.config)
        if decision:
            new_state = new_state.add_decision(decision)

    return new_state


def keyword_filter_agent(state: ProcessingState) -> ProcessingState:
    """Process emails through keyword filter.

    Pure function that checks each email for critical keywords.
    Only processes emails not yet decided.
    """
    new_state = state

    # Get UIDs that already have decisions
    decided_uids = {d.email.uid for d in state.decisions}

    for email in state.emails:
        # Skip if already processed or decided
        if email.uid in state.processed_uids or email.uid in decided_uids:
            continue

        decision = check_keywords(email, state.config)
        if decision:
            new_state = new_state.add_decision(decision)

    return new_state


def whitelist_filter_agent(state: ProcessingState) -> ProcessingState:
    """Process emails through whitelist filter.

    Pure function that checks each email against whitelisted domains.
    Only processes emails not yet decided.
    """
    new_state = state

    # Get UIDs that already have decisions
    decided_uids = {d.email.uid for d in state.decisions}

    for email in state.emails:
        # Skip if already processed or decided
        if email.uid in state.processed_uids or email.uid in decided_uids:
            continue

        decision = check_whitelist(email, state.config)
        if decision:
            new_state = new_state.add_decision(decision)

    return new_state


def sender_pattern_filter_agent(state: ProcessingState) -> ProcessingState:
    """Process emails through sender pattern filter.

    Pure function that checks sender history for auto-delete patterns.
    Only processes emails not yet decided.
    """
    if not state.config.enable_sender_tracking:
        return state

    new_state = state

    # Get UIDs that already have decisions
    decided_uids = {d.email.uid for d in state.decisions}

    for email in state.emails:
        # Skip if already processed or decided
        if email.uid in state.processed_uids or email.uid in decided_uids:
            continue

        sender_stats = state.sender_stats.get(
            email.sender, SenderStats(sender=email.sender)
        )
        decision = check_sender_pattern(email, sender_stats, state.config)
        if decision:
            new_state = new_state.add_decision(decision)

    return new_state


def ai_classifier_agent(state: ProcessingState) -> ProcessingState:
    """Process remaining emails through AI classifier.

    NOT a pure function (calls Ollama). Processes all emails that haven't
    been filtered by deterministic rules.
    """
    new_state = state

    # Get UIDs that already have decisions
    decided_uids = {d.email.uid for d in state.decisions}

    for email in state.emails:
        # Skip if already processed or decided
        if email.uid in state.processed_uids or email.uid in decided_uids:
            continue

        # This email needs AI classification
        decision = classify_with_ai(email, state.config)
        new_state = new_state.add_decision(decision)

    return new_state


def log_progress_agent(state: ProcessingState) -> ProcessingState:
    """Log current progress.

    Pure function (logging is read-only side effect).
    """
    print("\n=== Batch Progress ===")
    print(f"Emails in batch: {len(state.emails)}")
    print(f"Decisions made: {len(state.decisions)}")
    print(f"Total processed: {state.total_processed}")
    print(f"Total kept: {state.total_kept}")
    print(f"Total deleted: {state.total_deleted}")
    print(f"Consecutive empty batches: {state.consecutive_empty_batches}")

    if state.decisions:
        print("\n=== Decision Breakdown ===")
        reason_counts: dict[str, int] = {}
        for d in state.decisions:
            reason_counts[d.reason.value] = reason_counts.get(d.reason.value, 0) + 1
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")

    return state


def check_termination_agent(state: ProcessingState) -> ProcessingState:
    """Check if processing should terminate.

    Pure function that adds an error if termination condition met.
    """
    if state.consecutive_empty_batches >= 3:
        return state.add_error(
            "No new emails found in 3 consecutive batches. Terminating."
        )

    return state
