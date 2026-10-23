"""Deterministic sender/domain rules learned from classifier decisions.

The learning phase (``inbox-reaper learn``) is read-only: it classifies emails,
feeds every decision to a :class:`RuleLearner`, and writes the resulting
:class:`RuleSet` to a JSON file that can be reviewed and edited by hand.
``inbox-reaper process --rules`` then applies those rules before falling back
to heuristics and AI, so most mutations are driven by reviewable rules.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parseaddr
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .state import Decision, Email, EmailDecision, FilterReason

RULES_FILE_VERSION = 1

# Free-mail providers: many unrelated people share these domains, so a
# domain-wide rule would be wrong. Sender-level rules are still allowed.
SHARED_DOMAINS = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "msn.com",
        "yahoo.com",
        "icloud.com",
        "me.com",
        "aol.com",
        "proton.me",
        "protonmail.com",
    }
)

# Keep decisions from these filters are explicit user intent (attachments,
# keywords, whitelist) or strong transactional signals; any one of them vetoes
# learning a delete rule for that sender/domain.
_DETERMINISTIC_KEEP_REASONS = frozenset(
    {
        FilterReason.ATTACHMENT,
        FilterReason.KEYWORD,
        FilterReason.WHITELIST,
    }
)


class RuleMatch(str, Enum):
    """What part of the From address a rule matches."""

    SENDER = "sender"  # exact address, e.g. deals@e.shop.com
    DOMAIN = "domain"  # domain and its subdomains, e.g. shop.com


class Rule(BaseModel):
    """A single deterministic rule."""

    match: RuleMatch
    value: str
    action: Decision
    support: int = Field(default=0, description="Emails the rule was learned from")
    agreement: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Share of emails agreeing"
    )
    source: str = "learned"  # "learned" or "manual"


def parse_sender(sender: str) -> tuple[str, str]:
    """Return (lowercased address, lowercased domain) of a From header."""
    address = (parseaddr(sender)[1] or sender).strip(" <>").lower()
    return address, address.rpartition("@")[2]


class RuleSet(BaseModel):
    """An ordered collection of rules, serialized as a rules JSON file."""

    version: int = RULES_FILE_VERSION
    account: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)
    emails_observed: int = 0
    min_support: int = 0
    min_agreement: float = 0.0
    rules: list[Rule] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        self._senders = {
            r.value.lower(): r for r in self.rules if r.match == RuleMatch.SENDER
        }
        self._domains = {
            r.value.lower(): r for r in self.rules if r.match == RuleMatch.DOMAIN
        }

    def match(self, email: Email) -> Rule | None:
        """Find the most specific rule for an email.

        Precedence: exact sender, then the sender's domain, then each parent
        domain (mail.shop.com -> shop.com), never a bare TLD.
        """
        address, domain = parse_sender(email.sender)
        if address in self._senders:
            return self._senders[address]

        labels = domain.split(".")
        for i in range(len(labels) - 1):
            rule = self._domains.get(".".join(labels[i:]))
            if rule:
                return rule
        return None

    def decide(self, email: Email) -> EmailDecision | None:
        """Turn a matching rule into a decision for ``email``."""
        rule = self.match(email)
        if rule is None:
            return None
        return EmailDecision(
            email=email,
            decision=rule.action,
            reason=FilterReason.RULE,
            confidence=rule.agreement,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RuleSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("version", RULES_FILE_VERSION)
        if version != RULES_FILE_VERSION:
            raise ValueError(f"Unsupported rules file version {version} in {path}")
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


@dataclass
class _Tally:
    keep: int = 0
    delete: int = 0
    deterministic_keep: int = 0
    senders: set[str] = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.keep + self.delete


class RuleLearner:
    """Aggregates classifier decisions into sender and domain rules."""

    def __init__(self) -> None:
        self._by_sender: dict[str, _Tally] = defaultdict(_Tally)
        self._by_domain: dict[str, _Tally] = defaultdict(_Tally)
        self.observed = 0

    def observe(self, decision: EmailDecision) -> None:
        if decision.decision not in (Decision.KEEP, Decision.DELETE):
            return
        address, domain = parse_sender(decision.email.sender)
        if not domain:
            return

        self.observed += 1
        deterministic_keep = (
            decision.decision == Decision.KEEP
            and decision.reason in _DETERMINISTIC_KEEP_REASONS
        )
        for key, tallies in ((address, self._by_sender), (domain, self._by_domain)):
            tally = tallies[key]
            if decision.decision == Decision.DELETE:
                tally.delete += 1
            else:
                tally.keep += 1
            tally.deterministic_keep += deterministic_keep
            tally.senders.add(address)

    @staticmethod
    def _rule_for(
        match: RuleMatch,
        value: str,
        tally: _Tally,
        min_support: int,
        min_agreement: float,
    ) -> Rule | None:
        if tally.total < min_support:
            return None
        if tally.delete >= tally.keep:
            # A single deterministic keep vetoes a delete rule
            if tally.deterministic_keep:
                return None
            action, agreeing = Decision.DELETE, tally.delete
        else:
            action, agreeing = Decision.KEEP, tally.keep

        agreement = agreeing / tally.total
        if agreement < min_agreement:
            return None
        return Rule(
            match=match,
            value=value,
            action=action,
            support=tally.total,
            agreement=round(agreement, 3),
        )

    def build(
        self, min_support: int = 3, min_agreement: float = 0.9, account: str = ""
    ) -> RuleSet:
        """Produce rules for consistently classified senders and domains.

        Domain rules are only emitted when they generalize over at least two
        senders (otherwise the sender rule suffices), are not on a shared
        free-mail domain, and are not contradicted by a sender rule.
        """
        rules: list[Rule] = []
        sender_actions: dict[str, Decision] = {}

        for address, tally in self._by_sender.items():
            rule = self._rule_for(
                RuleMatch.SENDER, address, tally, min_support, min_agreement
            )
            if rule:
                rules.append(rule)
                sender_actions[address] = rule.action

        for domain, tally in self._by_domain.items():
            if domain in SHARED_DOMAINS or len(tally.senders) < 2:
                continue
            rule = self._rule_for(
                RuleMatch.DOMAIN, domain, tally, min_support, min_agreement
            )
            if rule is None:
                continue
            if any(
                sender_actions.get(s, rule.action) != rule.action for s in tally.senders
            ):
                continue
            rules.append(rule)

        # Most-supported first, so the file reads as a ranked review list
        rules.sort(key=lambda r: (-r.support, r.match.value, r.value))
        return RuleSet(
            account=account,
            emails_observed=self.observed,
            min_support=min_support,
            min_agreement=min_agreement,
            rules=rules,
        )
