"""Feature-vector pipeline: independent processors each append named features.

Every processor is a pure function ``(email, context) -> {name: value}``. Values
are floats, mostly 0/1 flags or scores in [0, 1], so a trained combiner
(:mod:`inbox_reaper.combiner`) can weigh them. :func:`extract_features` runs all
processors and prefixes each feature with its processor's name, e.g.
``header.list_id``. :func:`describe_features` turns the salient ones into a
short summary the AI classifier receives with the email.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.utils import getaddresses, parseaddr
from enum import StrEnum

from .agents import (
    check_attachments,
    check_keywords,
    check_marketing_indicators,
    check_transactional_patterns,
    check_whitelist,
    sender_domain,
)
from .rules import RuleSet
from .state import Config, Decision, Email, EmailDecision, FilterReason
from .text_utils import url_domain
from .timeutil import to_utc, utcnow

type FeatureVector = dict[str, float]


@dataclass(frozen=True)
class FeatureContext:
    """What processors may consult besides the email itself."""

    config: Config
    rules: RuleSet | None = None
    now: datetime = field(default_factory=utcnow)


type Processor = Callable[[Email, FeatureContext], FeatureVector]


def _flag(value: object) -> float:
    return 1.0 if value else 0.0


def _org(domain: str) -> str:
    """Crude registrable domain (last two labels): mail.chase.com -> chase.com."""
    return ".".join(domain.split(".")[-2:])


# ---------------------------------------------------------------------------
# Deterministic filters (the existing checks, as features instead of verdicts)
# ---------------------------------------------------------------------------

_FILTERS = {
    "attachment": check_attachments,
    "keyword": check_keywords,
    "whitelist": check_whitelist,
    "transactional": check_transactional_patterns,
    "marketing": check_marketing_indicators,
}


def filter_features(email: Email, ctx: FeatureContext) -> FeatureVector:
    out: FeatureVector = {}
    for name, check in _FILTERS.items():
        decision = check(email, ctx.config)
        out[f"{name}_keep"] = _flag(decision and decision.decision == Decision.KEEP)
        out[f"{name}_delete"] = _flag(decision and decision.decision == Decision.DELETE)
    return out


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

_ESP_HEADERS = (
    "x-sg-eid", "x-mailgun-tag", "x-mailgun-sid", "x-mc-user", "x-campaign",
    "x-campaignid", "x-sfmc-stack", "x-ses-outgoing", "x-mandrill-user",
    "x-msys-api", "x-cm-message-id", "x-marketo-message-id", "x-mailjet-campaign",
    "x-rpcampaign", "x-emarsys-identify", "x-ibm-mkt", "x-job",
)  # fmt: skip
_GITHUB_DIRECT = {"mention", "review_requested", "assign", "author", "team_mention"}
_DKIM_DOMAIN_RE = re.compile(r"\bd=([^;\s]+)")
_BCL_RE = re.compile(r"\bBCL:(\d)")
_REPLY_PREFIX_RE = re.compile(r"\s*(re|fw|fwd|aw|sv)\s*:", re.IGNORECASE)


def header_features(email: Email, ctx: FeatureContext) -> FeatureVector:
    h = email.headers
    from_domain = sender_domain(email.sender)
    auth = h.get("authentication-results", "").lower()
    dkim_domains = [
        d.lower() for d in _DKIM_DOMAIN_RE.findall(h.get("dkim-signature", ""))
    ]
    bcl = _BCL_RE.search(h.get("x-microsoft-antispam", ""))
    scl = h.get("x-ms-exchange-organization-scl", "").strip()
    reply_to = parseaddr(h.get("reply-to", ""))[1].rpartition("@")[2].lower()
    recipients = getaddresses([h.get("to", ""), h.get("cc", "")])
    precedence = h.get("precedence", "").strip().lower()
    auto_submitted = h.get("auto-submitted", "").strip().lower()
    github_reason = h.get("x-github-reason", "").strip().lower()
    return {
        "list_id": _flag("list-id" in h),
        "list_unsubscribe": _flag(
            email.has_list_unsubscribe or "list-unsubscribe" in h
        ),
        "one_click_unsubscribe": _flag("list-unsubscribe-post" in h),
        "precedence_bulk": _flag(precedence in {"bulk", "list", "junk"}),
        "auto_submitted": _flag(auto_submitted not in {"", "no"}),
        "feedback_id": _flag("feedback-id" in h),
        "esp": _flag(any(k in h for k in _ESP_HEADERS)),
        "reply_thread": _flag("in-reply-to" in h),
        "subject_reply": _flag(_REPLY_PREFIX_RE.match(email.subject)),
        "dkim": _flag(dkim_domains),
        "dkim_aligned": _flag(any(_org(d) == _org(from_domain) for d in dkim_domains)),
        "spf_pass": _flag("spf=pass" in auth),
        "dmarc_pass": _flag("dmarc=pass" in auth),
        "reply_to_elsewhere": _flag(reply_to and _org(reply_to) != _org(from_domain)),
        "many_recipients": _flag(len([a for _, a in recipients if a]) > 3),
        "bulk_complaint_level": int(bcl.group(1)) / 9 if bcl else 0.0,
        "spam_confidence": (int(scl) / 9) if scl.isdigit() else 0.0,
        "github": _flag("x-github-reason" in h),
        "github_direct": _flag(github_reason in _GITHUB_DIRECT),
    }


# ---------------------------------------------------------------------------
# Links and HTML structure
# ---------------------------------------------------------------------------

_TRACKING_RE = re.compile(
    r"(^|\.)(click|clicks|email|e|em|links?|trk|track|t|r|go|mail)\.|list-manage|"
    r"sendgrid\.net|mailchi\.mp|exacttarget|awstrack|/track|/click|/ls/click|"
    r"utm_(source|medium|campaign)",
    re.IGNORECASE,
)


def link_features(email: Email, ctx: FeatureContext) -> FeatureVector:
    links = email.links
    from_org = _org(sender_domain(email.sender))
    domains = [url_domain(u) for u in links]
    tracking = sum(bool(_TRACKING_RE.search(u)) for u in links)
    offsite = sum(_org(d) != from_org for d in domains)
    return {
        "count": min(len(links), 50) / 50,
        "tracking_share": tracking / len(links) if links else 0.0,
        "offsite_share": offsite / len(domains) if domains else 0.0,
        "images": min(email.image_count, 30) / 30,
        "tracking_pixel": _flag(email.tracking_pixels),
    }


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

_FREEMAIL = {
    "gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "live.com", "msn.com", "icloud.com", "me.com", "aol.com", "proton.me",
    "protonmail.com",
}  # fmt: skip
_AUTOMATED_LOCAL_RE = re.compile(
    r"no-?reply|do-?not-?reply|notifications?|notify|alerts?|news(letter)?|"
    r"mailer|info|updates?|digest|marketing|offers?|deals|promo|hello|team",
    re.IGNORECASE,
)


def sender_features(email: Email, ctx: FeatureContext) -> FeatureVector:
    name, address = parseaddr(email.sender)
    local, _, domain = address.lower().rpartition("@")
    return {
        "freemail": _flag(domain in _FREEMAIL),
        "automated_local_part": _flag(_AUTOMATED_LOCAL_RE.search(local)),
        "subdomain_sender": _flag(domain.count(".") >= 2),
        "has_display_name": _flag(name.strip()),
    }


# ---------------------------------------------------------------------------
# Learned rules
# ---------------------------------------------------------------------------


def rule_features(email: Email, ctx: FeatureContext) -> FeatureVector:
    rule = ctx.rules.match(email) if ctx.rules else None
    return {
        "keep": _flag(rule and rule.action == Decision.KEEP),
        "delete": _flag(rule and rule.action == Decision.DELETE),
    }


# ---------------------------------------------------------------------------
# Time sensitivity
# ---------------------------------------------------------------------------


class Lifetime(StrEnum):
    """How long an email stays useful to the recipient."""

    MINUTES = "minutes"  # one-time codes, sign-in links: useless after about an hour
    DAYS = "days"  # deliveries, reservations, sign-in alerts: useless after weeks
    FOREVER = "forever"  # records worth keeping, or not time-sensitive at all


# Fresh codes are kept; past the TTL they are expired. Between the two the
# processor reports age only and leaves the decision to other evidence.
FRESH_FOR = {Lifetime.MINUTES: timedelta(minutes=30)}
TTL = {Lifetime.MINUTES: timedelta(hours=1), Lifetime.DAYS: timedelta(days=30)}

_MINUTES_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(verification|security|one[- ]time|login|log[- ]in|sign[- ]in|access|"
        r"authentication|auth|2fa|mfa|otp|single[- ]use)\s+"
        r"(code|passcode|pin|password)\b",
        r"\bone[- ]time (password|passcode)\b",
        r"\byour code is\b|\bcode:\s*\d{4,8}\b|\b\d{4,8} is your\b",
        r"\b(magic|sign[- ]in|log[- ]?in) link\b",
        r"\b(verify|confirm) your (e-?mail|email address|account)\b",
        r"\breset your password\b|\bpassword reset (request|link|code)\b",
        r"\bgate (change|update|is now)\b",
        r"\bbattery (level )?is low\b|\blow battery\b",
        r"\binformed delivery\b|\bdaily digest\b",
    )
]
_DAYS_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bnew (sign[- ]?in|login|log[- ]in)\b|\bsigned in (to|from|on)\b|"
        r"\bnoticed a new\b",
        r"\bout for delivery\b|\barriving (today|tomorrow)\b|"
        r"\bhas (been )?delivered\b|\bwas delivered\b|\bdelivery attempt\b",
        r"\b(has|have) shipped\b|\bin transit\b|\bon (its|the) way\b|"
        r"\btracking number\b",
        r"\b(reservation|appointment|booking) (is )?"
        r"(confirmed|confirmation|reminder)\b",
        r"\bitinerary\b|\bcheck[- ]in (is )?(now )?open\b|\bboarding pass\b",
    )
]


def email_lifetime(email: Email) -> Lifetime:
    """Classify how long ``email`` stays useful from its subject and opening text."""
    text = f"{email.subject}\n{email.body[:600]}"
    if any(p.search(text) for p in _MINUTES_PATTERNS):
        return Lifetime.MINUTES
    if any(p.search(text) for p in _DAYS_PATTERNS):
        return Lifetime.DAYS
    return Lifetime.FOREVER


def email_age(email: Email, now: datetime) -> timedelta:
    return max(timedelta(0), now - to_utc(email.date, naive_is_local=False))


def temporal_features(email: Email, ctx: FeatureContext) -> FeatureVector:
    lifetime = email_lifetime(email)
    age = email_age(email, ctx.now)
    ttl = TTL.get(lifetime)
    fresh = FRESH_FOR.get(lifetime)
    return {
        "lifetime_minutes": _flag(lifetime == Lifetime.MINUTES),
        "lifetime_days": _flag(lifetime == Lifetime.DAYS),
        "expired": _flag(ttl is not None and age > ttl),
        "fresh": _flag(fresh is not None and age <= fresh),
        "log_age_days": math.log1p(age.total_seconds() / 86400) / math.log1p(3650),
    }


def check_expiry(
    email: Email, config: Config, now: datetime | None = None
) -> EmailDecision | None:
    """Keep fresh one-time codes; delete time-sensitive mail past its useful life.

    A code or sign-in link is kept for its first 30 minutes and deleted once it
    is over an hour old; delivery, reservation and sign-in notices are deleted
    after 30 days. Anything else, or a code 30-60 minutes old, is left to the
    remaining filters.
    """
    lifetime = email_lifetime(email)
    if lifetime == Lifetime.FOREVER:
        return None
    age = email_age(email, now or utcnow())
    if (fresh := FRESH_FOR.get(lifetime)) and age <= fresh:
        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.TIME_SENSITIVE,
            confidence=0.9,
        )
    if age > TTL[lifetime]:
        return EmailDecision(
            email=email,
            decision=Decision.DELETE,
            reason=FilterReason.EXPIRED,
            confidence=0.9,
        )
    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

PROCESSORS: dict[str, Processor] = {
    "filter": filter_features,
    "header": header_features,
    "link": link_features,
    "sender": sender_features,
    "rule": rule_features,
    "time": temporal_features,
}


def extract_features(
    email: Email,
    ctx: FeatureContext,
    processors: dict[str, Processor] = PROCESSORS,
) -> FeatureVector:
    """Run every processor and merge their features under ``<processor>.<name>``."""
    features: FeatureVector = {}
    for prefix, processor in processors.items():
        for name, value in processor(email, ctx).items():
            features[f"{prefix}.{name}"] = float(value)
    return features


# Human-readable summaries for the AI prompt, in priority order
_DESCRIPTIONS = {
    "time.expired": "time-sensitive and past its useful life",
    "time.fresh": "time-sensitive and still fresh",
    "rule.keep": "a learned rule keeps this sender",
    "rule.delete": "a learned rule deletes this sender",
    "header.precedence_bulk": "sent as bulk mail (Precedence header)",
    "header.list_id": "sent to a mailing list (List-Id)",
    "header.list_unsubscribe": "has an unsubscribe header",
    "header.esp": "sent through a bulk email platform",
    "header.github": "GitHub notification",
    "header.reply_thread": "a reply in a conversation",
    "header.dkim_aligned": "signed by the sender's own domain",
    "sender.freemail": "sent from a personal email provider",
    "sender.automated_local_part": "automated sender address",
    "filter.transactional_keep": "looks transactional (order, payment or receipt)",
    "link.tracking_pixel": "contains a tracking pixel",
}


def describe_features(features: FeatureVector, limit: int = 8) -> str:
    """Short semicolon-separated list of the signals that fired."""
    fired = [
        text for key, text in _DESCRIPTIONS.items() if features.get(key, 0.0) >= 0.5
    ]
    if features.get("link.tracking_share", 0.0) >= 0.5:
        fired.append("most links are tracking links")
    return "; ".join(fired[:limit])
