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


_PERSON_NAME_RE = re.compile(r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?")
_VIA_RE = re.compile(r"\bvia\b", re.IGNORECASE)


def _is_person(name: str, local: str) -> bool:
    """'Matthew Prince' <mprince@...>: two or three name-like words, one of
    which (three letters or more) appears in the address."""
    words = name.replace('"', "").split()
    if not 2 <= len(words) <= 3 or not all(_PERSON_NAME_RE.fullmatch(w) for w in words):
        return False
    local = re.sub(r"[^a-z]", "", local)
    return any(len(w) >= 3 and w.lower() in local for w in words)


def sender_features(email: Email, ctx: FeatureContext) -> FeatureVector:
    name, address = parseaddr(email.sender)
    local, _, domain = address.lower().rpartition("@")
    return {
        "freemail": _flag(domain in _FREEMAIL),
        "automated_local_part": _flag(_AUTOMATED_LOCAL_RE.search(local)),
        "subdomain_sender": _flag(domain.count(".") >= 2),
        "has_display_name": _flag(name.strip()),
        # A person writing, not a brand ("Zachary Gerard <zach-gerard@...>")
        "person": _flag(_is_person(name, local)),
        # A person through a service ("Vilas via Dropbox", "Anthony via RT")
        "via_service": _flag(_VIA_RE.search(name)),
    }


# ---------------------------------------------------------------------------
# Content: records versus promotions, identifiers, personalisation, structure
# ---------------------------------------------------------------------------

# Words that make an email a record worth keeping, even when sent in bulk
_RECORD_SUBJECT_RE = re.compile(
    r"\b(receipts?|statements?|invoices?|orders?|reservations?|bookings?|"
    r"itinerar(y|ies)|redemptions?|applications?|tax|1099|w-?2|polic(y|ies)|"
    r"claims?|refunds?|payments?|bills?|deposits?|withdrawals?|transfers?|"
    r"confirm(ed|ation)|cancel(l)?ed|contracts?|lease|documents?|"
    r"tickets? (for|to)|e-?tickets?)\b",
    re.IGNORECASE,
)
_RECORD_BODY_RE = re.compile(
    r"\b(order (number|total|summary|date)|total (charged|paid|amount)|"
    r"amount (due|paid|charged)|subtotal|account (number|ending)|"
    r"transaction (id|date|details)|payment method|billing address|"
    r"confirmation (number|code)|reservation (number|code)|"
    r"(available|ready) to view|view your statement)\b",
    re.IGNORECASE,
)
_PROMO_RE = re.compile(
    r"\d+\s?% off|\bsale\b|\bdeals?\b|\bsave( up to)?\b|limited[- ]time|"
    r"last chance|ends (today|tonight|soon|sunday|midnight)|free shipping|"
    r"\bexclusive\b|new arrivals|don'?t miss|\bshop (now|the)\b|"
    r"\b(special )?offers?\b|\bcoupons?\b|promo code|\bfree\b|"
    r"best[- ]sellers?|\bjust dropped\b|\bhurry\b|\btreat yourself\b",
    re.IGNORECASE,
)
_PRICE_RE = re.compile(r"[$£€₹]\s?\d|\d+\s?%")
# Amounts with cents ($42.10) appear on receipts; promotions say "$20 off"
_AMOUNT_RE = re.compile(r"[$£€₹]\s?\d[\d,]*\.\d{2}\b|\b(USD|INR|EUR)\s?\d")
# Order/booking numbers; not bare "#1234", which is mostly GitHub issue numbers
_IDENTIFIER_RE = re.compile(
    r"\b(order|confirmation|reservation|booking|invoice|receipt|record locator)"
    r"\s*(#|no\.?|number|code|id)?\s*[:#]?\s*(?=[A-Z0-9-]*\d)[A-Z0-9-]{5,}\b",
    re.IGNORECASE,
)
_ACCOUNT_ENDING_RE = re.compile(
    r"\bend(ing|s) in:?\s*\d{4}\b|[*xX•]{2,}\s?\d{4}\b", re.IGNORECASE
)
# "Dear customer"; not "Hi there", which real account notices use too
_GENERIC_GREETING_RE = re.compile(
    r"\b(dear|hi|hello)\s+(valued\s+)?(customer|member|client|guest|"
    r"subscriber|shopper|traveler|user|cardmember)\b",
    re.IGNORECASE,
)
_SHARED_RE = re.compile(r"\bshared\b.*\bwith you\b|\binvited you\b", re.IGNORECASE)
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff]")
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")
_WORD_RE = re.compile(r"\w+")


def user_names(email: Email, config: Config) -> list[str]:
    """The mailbox owner's name words: from the To/Cc display name for their
    own address when shown, else from the address's local part."""
    me = config.email.lower()
    for name, address in getaddresses(
        [email.headers.get("to", ""), email.headers.get("cc", "")]
    ):
        if me and address.lower() == me and name:
            return [w for w in re.findall(r"[^\W\d_]+", name) if len(w) >= 3]
    # Fall back to the address itself: jane.doe@... -> jane, doe
    return [w for w in re.findall(r"[a-z]+", me.partition("@")[0]) if len(w) >= 4]


def _addressed_to_me(email: Email, config: Config) -> bool:
    me = config.email.lower()
    to = getaddresses([email.headers.get("to", "")])
    return bool(me) and any(address.lower() == me for _, address in to)


def content_features(email: Email, ctx: FeatureContext) -> FeatureVector:
    subject, body = email.subject, email.body
    names = user_names(email, ctx.config)
    opening = body[:300]
    name_re = (
        re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE)
        if names
        else None
    )
    words = len(_WORD_RE.findall(body))
    caps = [w for w in _CAPS_WORD_RE.findall(subject) if len(w) >= 3]
    return {
        # Records versus promotions
        "record_subject": _flag(_RECORD_SUBJECT_RE.search(subject)),
        "record_body": min(len(_RECORD_BODY_RE.findall(body)), 3) / 3,
        "promo_subject": _flag(_PROMO_RE.search(subject)),
        "promo_body": min(len(_PROMO_RE.findall(body)), 5) / 5,
        "price_subject": _flag(_PRICE_RE.search(subject)),
        "emoji_subject": _flag(_EMOJI_RE.search(subject)),
        "caps_subject": _flag(len(caps) >= 2),
        "exclaim_subject": _flag("!" in subject),
        # Identifiers that records carry
        "identifier": _flag(_IDENTIFIER_RE.search(f"{subject}\n{body}")),
        "account_ending": _flag(_ACCOUNT_ENDING_RE.search(f"{subject}\n{body}")),
        "amount": _flag(_AMOUNT_RE.search(f"{subject}\n{body}")),
        # Personalisation
        "addressed_to_me": _flag(_addressed_to_me(email, ctx.config)),
        "name_subject": _flag(name_re and name_re.search(subject)),
        "name_greeting": _flag(name_re and name_re.search(opening)),
        "generic_greeting": _flag(_GENERIC_GREETING_RE.search(opening)),
        "shared_with_you": _flag(_SHARED_RE.search(subject)),
        # Structure: a promotion is mostly layout, a record mostly content
        "words": math.log1p(words) / math.log1p(5000),
        "link_density": min(len(email.links) / max(words, 1) * 100, 10) / 10,
        "image_density": min(email.image_count / max(words, 1) * 100, 10) / 10,
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
    """Keep fresh one-time codes and sign-in links; delete them once expired.

    A code or sign-in link is kept for its first 30 minutes and deleted once
    it is over an hour old (100% precise on a 1,025-email labelled sample).
    Day-scale mail (deliveries, reservations, sign-in alerts) is never deleted
    here: its patterns are broader, so its age only feeds the combiner as a
    feature. A code 30-60 minutes old is left to the remaining filters.
    """
    if email_lifetime(email) != Lifetime.MINUTES:
        return None
    age = email_age(email, now or utcnow())
    if age <= FRESH_FOR[Lifetime.MINUTES]:
        return EmailDecision(
            email=email,
            decision=Decision.KEEP,
            reason=FilterReason.TIME_SENSITIVE,
            confidence=0.9,
        )
    if age > TTL[Lifetime.MINUTES]:
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
    "content": content_features,
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
    "content.record_subject": "subject names a record (receipt, statement, order...)",
    "content.identifier": "carries an order, booking or reference number",
    "content.account_ending": "mentions an account or card ending",
    "content.name_subject": "addresses you by name in the subject",
    "content.name_greeting": "greets you by name",
    "content.promo_subject": "promotional subject (sale, % off, offer...)",
    "content.generic_greeting": "generic greeting (dear customer...)",
    "sender.person": "written by a person, not a brand",
    "sender.via_service": "shared by a person through a service",
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
