"""Agent functions for email classification.

Each check is a pure function (email, config) -> EmailDecision | None; the AI
classifier is the one function with a side effect (MLX inference).
"""

import json
import logging
import re
from email.utils import parseaddr

from pydantic import BaseModel, Field

from .mlx_backend import generate_text, generate_text_batch
from .state import (
    Config,
    Decision,
    Email,
    EmailDecision,
    FilterReason,
)
from .text_utils import (  # noqa: F401 - re-exported for callers/tests
    HTMLTextExtractor,
    remove_base64_images,
    strip_html,
    truncate_symmetric,
)


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

# Generous buffer for models that add explanatory text around the JSON
AI_MAX_TOKENS = 200

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
        for entry in config.whitelist_domains:
            whitelisted = entry.lower()
            if domain == whitelisted or domain.endswith("." + whitelisted):
                return EmailDecision(
                    email=email,
                    decision=Decision.KEEP,
                    reason=FilterReason.WHITELIST,
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


def parse_ai_response(response: str) -> AIClassificationResponse:
    """Extract the classification JSON from a model response.

    Tolerates markdown code fences and text around the JSON object.

    Raises:
        ValueError: If no valid classification JSON is present
    """
    response_text = response.strip()

    # Remove markdown code fences if present (```json ... ``` or ``` ... ```)
    if response_text.startswith("```"):
        # Drop the opening fence line (if any), then the closing fence
        _, newline, rest = response_text.partition("\n")
        if newline:
            response_text = rest
        response_text = response_text.removesuffix("```").strip()

    # Find JSON object in response (handles cases where model adds text)
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        msg = (
            f"No valid JSON in response. "
            f"Response length: {len(response_text)}, "
            f"First 200 chars: '{response_text[:200]}'"
        )
        raise ValueError(msg)

    return AIClassificationResponse(**json.loads(response_text[json_start:json_end]))


def ai_messages(email: Email, signals: str = "") -> list[dict[str, str]]:
    """Chat messages asking the model to classify ``email``.

    ``signals`` is a short summary of what the automated checks found (see
    :func:`inbox_reaper.features.describe_features`); it is shown as context.
    """
    signal_line = f"Automated checks: {signals}\n\n" if signals else ""
    # Strip HTML and truncate symmetrically to preserve footer (unsubscribe links, etc.)
    body_preview = truncate_symmetric(email.body)

    # Use chat template for better JSON compliance with instruct models
    return [
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

{signal_line}Analyze whether this is marketing/promotional that can be safely deleted.
Output format: {{"is_marketing": true, "confidence": 0.95}}""",
        },
    ]


def _ai_keep(email: Email) -> EmailDecision:
    """Safe default when the model fails or its output can't be parsed."""
    return EmailDecision(
        email=email,
        decision=Decision.KEEP,
        reason=FilterReason.AI_CLASSIFIED,
        confidence=0.0,
    )


def ai_decision(email: Email, response: str, config: Config) -> EmailDecision:
    """Turn a model response into a decision; unparseable output keeps."""
    try:
        classification = parse_ai_response(response)
    except Exception as e:
        # Malformed model output is expected occasionally; log without traceback
        logger.error(  # noqa: TRY400
            "MLX classification failed for email %s: %s", email.uid, e
        )
        return _ai_keep(email)

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


def classify_with_ai(email: Email, config: Config, signals: str = "") -> EmailDecision:
    """Classify one email with the MLX model; KEEP on any failure."""
    try:
        response = generate_text(
            model_name=config.model_name,
            messages=ai_messages(email, signals),
            max_tokens=AI_MAX_TOKENS,
        )
    except Exception as e:
        logger.error(  # noqa: TRY400
            "MLX classification failed for email %s: %s", email.uid, e
        )
        return _ai_keep(email)
    return ai_decision(email, response, config)


def classify_batch_with_ai(
    emails: list[Email], config: Config, signals: list[str] | None = None
) -> list[EmailDecision]:
    """Classify several emails in one batched MLX call, in input order.

    Batching amortizes decoding across emails. If the whole call fails every
    email is kept; a single unparseable response keeps just that email.
    """
    if not emails:
        return []
    if len(emails) == 1:
        # Plain generation is ~10% faster than a batch of one
        return [classify_with_ai(emails[0], config, signals[0] if signals else "")]
    try:
        responses = generate_text_batch(
            model_name=config.model_name,
            messages_list=[
                ai_messages(email, sig)
                for email, sig in zip(
                    emails, signals or [""] * len(emails), strict=True
                )
            ],
            max_tokens=AI_MAX_TOKENS,
        )
    except Exception as e:
        logger.error(  # noqa: TRY400
            "MLX batch classification failed for %s emails: %s", len(emails), e
        )
        return [_ai_keep(email) for email in emails]
    return [
        ai_decision(email, response, config)
        for email, response in zip(emails, responses, strict=True)
    ]
