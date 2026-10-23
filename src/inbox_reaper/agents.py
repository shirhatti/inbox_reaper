"""Agent functions for email classification.

All agents are pure functions: ProcessingState -> ProcessingState
Each agent processes emails in the current batch and adds decisions.
"""

import json
import logging
import re
from email.utils import parseaddr
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from .mlx_backend import generate_text
from .state import (
    Config,
    Decision,
    Email,
    EmailDecision,
    FilterReason,
    SenderStats,
)
from .text_utils import (  # noqa: F401 - re-exported for callers/tests
    HTMLTextExtractor,
    remove_base64_images,
    strip_html,
    truncate_symmetric,
)

if TYPE_CHECKING:
    from .ui.events import EventEmitter


class AIClassificationResponse(BaseModel):
    """Schema for AI classification response with confidence score."""

    is_marketing: bool = Field(
        description="Whether the email is marketing/promotional content"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for the classification",
    )


# Cache the JSON schema at module level (generated once, reused for all classifications)
_AI_CLASSIFICATION_SCHEMA = AIClassificationResponse.model_json_schema()

logger = logging.getLogger(__name__)

# Deterministic filter patterns, compiled once rather than on every email
_ORDER_NUMBER_RE = re.compile(r"(receipt|invoice|order|confirmation)\s*#?\s*\d+")
_SENDER_DOMAIN_RE = re.compile(r"@([^\s>]+)")

_AMAZON_TRANSACTIONAL_SENDERS = (
    "ship-confirm@amazon.com",
    "auto-confirm@amazon.com",
    "digital-noreply@amazon.com",
    "order-update@amazon.com",
)
_PAYMENT_KEYWORDS = (
    "thank you for your payment",
    "payment received",
    "payment confirmation",
    "your package is arriving",
    "has shipped",
    "order confirmed",
    "out for delivery",
    "delivered to",
)
_PERSONAL_DOMAINS = ("@gmail.com", "@outlook.com", "@hotmail.com", "@yahoo.com")
_MARKETING_SUBDOMAIN_PREFIXES = (
    "e.",
    "email.",
    "click.",
    "view.",
    "news.",
    "marketing.",
)
_EMAIL_SERVICE_PLATFORMS = (
    "sendgrid",
    "mailgun",
    "mailchimp",
    "constantcontact",
    "exacttarget",
)


def sender_domain(sender: str) -> str:
    """Return the lowercased domain of a From header like 'Name <a@b.com>'."""
    address = parseaddr(sender)[1] or sender
    return address.rpartition("@")[2].strip(" <>").lower()


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
    # Matches the domain itself and any subdomain (alerts.bank.com for bank.com)
    if "@" in email.sender:
        domain = sender_domain(email.sender)
        for whitelisted in config.whitelist_domains:
            whitelisted = whitelisted.lower()
            if domain == whitelisted or domain.endswith("." + whitelisted):
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


def check_transactional_patterns(email: Email, config: Config) -> EmailDecision | None:
    """Check if email matches transactional patterns.

    Pure function that returns a KEEP decision if transactional indicators found.
    Catches receipts, order confirmations, shipping, and personal emails.
    """
    text = f"{email.subject} {email.body}".lower()
    subject_lower = email.subject.lower()
    sender_lower = email.sender.lower()

    # Pattern 1: Receipt/Invoice/Order numbers in subject
    if _ORDER_NUMBER_RE.search(subject_lower):
        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.KEYWORD,
            confidence=0.98,
        )

    # Pattern 2: Transactional Amazon senders
    if any(sender in sender_lower for sender in _AMAZON_TRANSACTIONAL_SENDERS):
        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.WHITELIST,
            confidence=0.95,
        )

    # Pattern 3: Payment keywords
    for keyword in _PAYMENT_KEYWORDS:
        if keyword in text:
            return EmailDecision(
                email=email,
                decision=Decision.KEEP,
                reason=FilterReason.KEYWORD,
                confidence=0.90,
            )

    # Pattern 4: Personal emails (gmail, outlook from individuals)
    # ...unless automated ("noreply", "no-reply")
    if (
        any(domain in sender_lower for domain in _PERSONAL_DOMAINS)
        and "noreply" not in sender_lower
        and "no-reply" not in sender_lower
    ):
        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.WHITELIST,
            confidence=0.85,
        )

    return None


def check_marketing_indicators(email: Email, config: Config) -> EmailDecision | None:
    """Check if email has marketing indicators in metadata.

    Pure function that returns a DELETE decision if marketing signals found:
    unsubscribe links, "view in browser", and tracking subdomains.
    """
    body_lower = email.body.lower()
    sender_lower = email.sender.lower()

    # Strong marketing signals
    marketing_score = 0

    # Signal 1: Unsubscribe link (very strong)
    if "unsubscribe" in body_lower:
        marketing_score += 3

    # Signal 2: "View in browser" link
    if ("view" in body_lower and "browser" in body_lower) or (
        "click here" in body_lower and "web browser" in body_lower
    ):
        marketing_score += 2

    # Signal 3: Marketing platform subdomain
    sender_match = _SENDER_DOMAIN_RE.search(sender_lower)
    if sender_match:
        domain = sender_match.group(1)

        # e.domain.com or email.domain.com patterns
        if domain.startswith(_MARKETING_SUBDOMAIN_PREFIXES):
            marketing_score += 2

        # Generic email service platforms
        if any(service in domain for service in _EMAIL_SERVICE_PLATFORMS):
            marketing_score += 2

    # Signal 4: List-Unsubscribe header (bulk-mail marker set by mailing platforms)
    if (
        email.has_list_unsubscribe
        or "list-unsubscribe" in body_lower
        or "list-id" in body_lower
    ):
        marketing_score += 2

    # If score >= 3, it's definitely marketing
    if marketing_score >= 3:
        return EmailDecision(
            email=email,
            decision=Decision.DELETE,
            reason=FilterReason.SENDER_PATTERN,
            confidence=min(0.95, 0.70 + (marketing_score * 0.05)),
        )

    return None


def classify_with_ai(
    email: Email,
    config: Config,
    event_emitter: Optional["EventEmitter"] = None,
) -> EmailDecision:
    """Classify email using MLX LLM with structured JSON output.

    This is the only non-pure function (has side effect of calling MLX).
    Returns DELETE decision for marketing, KEEP for everything else.
    The model returns both classification and confidence score.
    """
    # Strip HTML and truncate symmetrically to preserve footer (unsubscribe links, etc.)
    body_preview = truncate_symmetric(email.body)

    # Use chat template for better JSON compliance with instruct models
    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON-only API. You MUST respond with ONLY valid JSON. "
                "No explanations, no reasoning, no markdown - just the JSON object."
            ),
        },
        {
            "role": "user",
            "content": f"""Classify this email as marketing/promotional or important.

KEEP if:
- Receipt: "Thank you for your payment (Receipt# 123)", "Invoice #456"
- Shipping: "Your package is arriving", "Order shipped", "Tracking number"
- Financial: "We mailed your card", "Statement available", "Account alert"
- Personal: Email from a person (not a company)

DELETE if:
- Marketing: Sales, discounts, coupons, "Save 20%", "Limited time"
- Newsletters: Updates, blog posts, curated content
- Notifications: GitHub, Slack, automated alerts
- Surveys: "Tell us what you think"

Email:
Subject: {email.subject}
From: {email.sender}
Body: {body_preview}

Analyze whether this is marketing/promotional that can be safely deleted.
Output format: {{"is_marketing": true, "confidence": 0.95}}""",
        },
    ]

    try:
        # Use MLX for inference with chat template
        response = generate_text(
            model_name=config.model_name,
            messages=messages,
            max_tokens=200,  # Generous buffer for models that add explanatory text
            event_emitter=event_emitter,
        )

        # Parse JSON response
        # Try to extract JSON from response (in case model adds extra text or markdown)
        response_text = response.strip()

        # Remove markdown code fences if present (```json ... ``` or ``` ... ```)
        if response_text.startswith("```"):
            # Find the end of opening fence
            first_newline = response_text.find("\n")
            if first_newline != -1:
                # Remove opening fence
                response_text = response_text[first_newline + 1 :]
            # Remove closing fence
            response_text = response_text.removesuffix("```")
            response_text = response_text.strip()

        # Find JSON object in response (handles cases where model adds text)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            parsed = json.loads(json_text)
            classification = AIClassificationResponse(**parsed)

            # Only delete if marked as marketing AND confidence >= threshold
            # Otherwise, keep (safe default for low confidence)
            if (
                classification.is_marketing
                and classification.confidence >= config.ai_confidence_threshold
            ):
                decision = Decision.DELETE
            else:
                decision = Decision.KEEP

            return EmailDecision(
                email=email,
                decision=decision,
                reason=FilterReason.AI_CLASSIFIED,
                confidence=classification.confidence,
            )
        # No valid JSON found
        raise ValueError(
            f"No valid JSON in response. "
            f"Response length: {len(response_text)}, "
            f"First 200 chars: '{response_text[:200]}'"
        )

    except Exception as e:
        # On error, default to KEEP (safe default)
        error_msg = f"MLX classification failed for email {email.uid}: {e}"

        if event_emitter:
            from .ui.events import EventType

            event_emitter.emit(
                EventType.PROCESSING_ERROR,
                error_msg,
                email_uid=email.uid,
                error=str(e),
            )
        else:
            # Malformed model output is expected occasionally; no traceback
            logger.error(error_msg)  # noqa: TRY400

        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.AI_CLASSIFIED,
            confidence=0.0,
        )
