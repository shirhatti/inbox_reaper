"""Agent functions for email classification.

All agents are pure functions: ProcessingState -> ProcessingState
Each agent processes emails in the current batch and adds decisions.
"""

import html
import json
import re
from html.parser import HTMLParser
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


def check_transactional_patterns(email: Email, config: Config) -> EmailDecision | None:
    """Check if email matches transactional patterns.

    Pure function that returns a KEEP decision if transactional indicators found.
    Catches receipts, order confirmations, shipping, and personal emails.
    """
    text = f"{email.subject} {email.body}".lower()
    subject_lower = email.subject.lower()
    sender_lower = email.sender.lower()

    import re

    # Pattern 1: Receipt/Invoice/Order numbers in subject
    if re.search(r"(receipt|invoice|order|confirmation)\s*#?\s*\d+", subject_lower):
        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.KEYWORD,
            confidence=0.98,
        )

    # Pattern 2: Transactional Amazon senders
    amazon_transactional = [
        "ship-confirm@amazon.com",
        "auto-confirm@amazon.com",
        "digital-noreply@amazon.com",
        "order-update@amazon.com",
    ]
    if any(sender in sender_lower for sender in amazon_transactional):
        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.WHITELIST,
            confidence=0.95,
        )

    # Pattern 3: Payment keywords
    payment_keywords = [
        "thank you for your payment",
        "payment received",
        "payment confirmation",
        "your package is arriving",
        "has shipped",
        "order confirmed",
        "out for delivery",
        "delivered to",
    ]
    for keyword in payment_keywords:
        if keyword in text:
            return EmailDecision(
                email=email,
                decision=Decision.KEEP,
                reason=FilterReason.KEYWORD,
                confidence=0.90,
            )

    # Pattern 4: Personal emails (gmail, outlook from individuals)
    personal_domains = ["@gmail.com", "@outlook.com", "@hotmail.com", "@yahoo.com"]
    if any(domain in sender_lower for domain in personal_domains):
        # Check if it's not automated (no "noreply", "no-reply")
        if "noreply" not in sender_lower and "no-reply" not in sender_lower:
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
    import re

    sender_match = re.search(r"@([^\s>]+)", sender_lower)
    if sender_match:
        domain = sender_match.group(1)

        # e.domain.com or email.domain.com patterns
        if (
            domain.startswith("e.")
            or domain.startswith("email.")
            or domain.startswith("click.")
            or domain.startswith("view.")
            or domain.startswith("news.")
            or domain.startswith("marketing.")
        ):
            marketing_score += 2

        # Generic email service platforms
        if any(
            service in domain
            for service in [
                "sendgrid",
                "mailgun",
                "mailchimp",
                "constantcontact",
                "exacttarget",
            ]
        ):
            marketing_score += 2

    # Signal 4: List-Unsubscribe header (if present in body)
    if "list-unsubscribe" in body_lower or "list-id" in body_lower:
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


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, ignoring tags and scripts."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_content = False

    def handle_starttag(self, tag, attrs):
        # Skip content in script and style tags
        if tag.lower() in ("script", "style"):
            self.skip_content = True

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style"):
            self.skip_content = False

    def handle_data(self, data):
        if not self.skip_content:
            self.text_parts.append(data)

    def get_text(self) -> str:
        return "".join(self.text_parts)


def remove_base64_images(text: str) -> str:
    """Remove base64 encoded images from text.

    Removes data URLs containing base64 encoded images which can appear in
    HTML emails as inline images. These bloat the text and provide no value
    for classification.

    Args:
        text: The text that may contain base64 encoded images

    Returns:
        Text with base64 image data removed
    """
    # Match data URLs for images: data:image/[type];base64,[base64data]
    # The base64 data consists of A-Za-z0-9+/= characters
    # We match until we hit a quote, whitespace, or < (end of attribute value)
    pattern = r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+"
    return re.sub(pattern, "", text)


def strip_html(text: str) -> str:
    """Strip HTML tags and extract plain text content.

    Removes HTML tags, scripts, styles, base64 images, and converts HTML
    entities to text. Cleans up excessive whitespace while preserving
    paragraph breaks.

    Args:
        text: The HTML or plain text to process

    Returns:
        Plain text with HTML removed and whitespace normalized
    """
    # Remove base64 encoded images first (they bloat the text)
    text = remove_base64_images(text)

    # If text doesn't contain HTML tags, return as-is
    if "<" not in text:
        return text

    # Parse HTML and extract text
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(text)
        plain_text = extractor.get_text()
    except Exception:
        # If HTML parsing fails, fall back to regex stripping
        plain_text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities (e.g., &nbsp; -> space, &lt; -> <)
    plain_text = html.unescape(plain_text)

    # Normalize whitespace: replace multiple spaces/tabs with single space
    plain_text = re.sub(r"[ \t]+", " ", plain_text)

    # Preserve paragraph breaks but remove excessive newlines (more than 2)
    plain_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", plain_text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in plain_text.split("\n")]
    plain_text = "\n".join(lines)

    return plain_text.strip()


def truncate_symmetric(text: str, max_length: int = 2000) -> str:
    """Truncate text symmetrically, keeping start and end.

    Strips HTML markup first to extract only meaningful text content,
    then preserves the beginning and end of the text (which often contains
    important footer information like unsubscribe links), removing the
    middle portion if the text exceeds max_length.

    Args:
        text: The text to truncate (HTML or plain text)
        max_length: Maximum length of the result (default: 2000)

    Returns:
        Truncated plain text with start and end preserved
    """
    # Strip HTML first to get clean text
    clean_text = strip_html(text)

    if len(clean_text) <= max_length:
        return clean_text

    # Reserve space for the truncation marker
    marker = "\n\n[... content truncated ...]\n\n"
    available = max_length - len(marker)

    # Split available space evenly between start and end
    start_length = available // 2
    end_length = available - start_length

    start = clean_text[:start_length]
    end = clean_text[-end_length:]

    return start + marker + end


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
            if response_text.endswith("```"):
                response_text = response_text[: -len("```")]
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
        else:
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
            import logging

            logging.getLogger(__name__).error(error_msg)

        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.AI_CLASSIFIED,
            confidence=0.0,
        )
