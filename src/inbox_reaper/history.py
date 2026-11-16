"""Mailbox history: how you have treated each sender, gathered read-only.

The history is built from the INBOX (sender, date and read/flagged state of
every message, via ``BODY.PEEK`` so nothing is marked read) and the Sent
folder (who you have written to and the Message-IDs of what you sent, opened
with EXAMINE so it cannot change). The ``history`` feature processor turns it
into per-email signals: have you read this sender's mail before, have you ever
written to them, is this a thread you took part in.

A history file holds addresses and message IDs from your mailbox, so keep it
out of version control like samples and labels.
"""

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from email.utils import getaddresses, parseaddr
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .timeutil import utcnow

if TYPE_CHECKING:  # the feature processors import this module without IMAP
    from .imap_client import AsyncIMAPClient

logger = logging.getLogger(__name__)

HISTORY_VERSION = 1
# UIDs per header FETCH: headers are small, so batches can be large
HISTORY_FETCH_BATCH = 500
_MESSAGE_ID_RE = re.compile(r"<[^<>\s]+>")


class InboxMessage(BaseModel):
    """One INBOX message as the history sees it."""

    sender: str  # lowercased address
    date: datetime
    seen: bool = False
    flagged: bool = False


class SenderStats(BaseModel):
    """Aggregate over the INBOX messages from one sender (or domain)."""

    count: int = 0
    seen: int = 0
    flagged: int = 0
    first: datetime | None = None
    last: datetime | None = None

    def add(self, message: InboxMessage) -> None:
        self.count += 1
        self.seen += message.seen
        self.flagged += message.flagged
        self.first = min(self.first or message.date, message.date)
        self.last = max(self.last or message.date, message.date)


class MailboxHistory(BaseModel):
    """Read-only snapshot of INBOX behaviour and Sent mail."""

    version: int = HISTORY_VERSION
    built_at: datetime = Field(default_factory=utcnow)
    # INBOX UID -> message
    messages: dict[str, InboxMessage] = Field(default_factory=dict)
    # Lowercased address -> times you sent mail to it (To, Cc or Bcc)
    contacted: dict[str, int] = Field(default_factory=dict)
    # Message-IDs of mail you sent (to recognise threads you took part in)
    sent_ids: list[str] = Field(default_factory=list)
    # For incremental updates: folder UIDVALIDITY and Sent UIDs already read
    inbox_validity: str | None = None
    sent_validity: str | None = None
    sent_uids: list[str] = Field(default_factory=list)

    @cached_property
    def by_sender(self) -> dict[str, SenderStats]:
        stats: dict[str, SenderStats] = {}
        for message in self.messages.values():
            stats.setdefault(message.sender, SenderStats()).add(message)
        return stats

    @cached_property
    def by_domain(self) -> dict[str, SenderStats]:
        stats: dict[str, SenderStats] = {}
        for message in self.messages.values():
            stats.setdefault(org_domain(message.sender), SenderStats()).add(message)
        return stats

    @cached_property
    def contacted_domains(self) -> frozenset[str]:
        return frozenset(org_domain(a) for a in self.contacted)

    @cached_property
    def sent_id_set(self) -> frozenset[str]:
        return frozenset(self.sent_ids)

    @classmethod
    def load(cls, path: str | Path) -> "MailboxHistory":
        history = cls.model_validate_json(Path(path).read_text())
        if history.version != HISTORY_VERSION:
            raise ValueError(
                f"{path}: history version {history.version}, expected "
                f"{HISTORY_VERSION}; rebuild it with 'inbox-reaper history'"
            )
        return history

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json())


def address_of(value: str) -> str:
    """Lowercased address from a From-style header value."""
    return parseaddr(value)[1].lower()


def org_domain(address: str) -> str:
    """Crude registrable domain of an address: a@mail.chase.com -> chase.com."""
    domain = address.rpartition("@")[2]
    return ".".join(domain.split(".")[-2:])


def message_ids(*values: str) -> list[str]:
    """Message-IDs referenced in header values (References, In-Reply-To)."""
    return [m for v in values for m in _MESSAGE_ID_RE.findall(v)]


def _batches(uids: list[str]) -> Iterable[list[str]]:
    for i in range(0, len(uids), HISTORY_FETCH_BATCH):
        yield uids[i : i + HISTORY_FETCH_BATCH]


async def build_history(
    client: "AsyncIMAPClient",
    *,
    previous: MailboxHistory | None = None,
    include_sent: bool = True,
) -> MailboxHistory:
    """Scan INBOX and the Sent folder, read-only, into a MailboxHistory.

    With ``previous`` (an earlier history of the same mailbox), only messages
    it does not already hold are fetched; read and flagged state always comes
    fresh from ``UID SEARCH SEEN`` / ``FLAGGED``, which are cheap. A changed
    UIDVALIDITY means the server renumbered the folder: it is rebuilt.

    Leaves the client with INBOX selected again.
    """
    from .imap_client import _parse_date  # noqa: PLC0415

    history = MailboxHistory()
    try:
        validity = await client.examine_mailbox("INBOX")
        known: dict[str, InboxMessage] = {}
        if previous is not None and previous.inbox_validity in (None, validity):
            known = previous.messages
        history.inbox_validity = validity
        uids = await client.search_uids("ALL")
        seen = set(await client.search_uids("SEEN"))
        flagged = set(await client.search_uids("FLAGGED"))
        new = [u for u in uids if u not in known]
        fetched: dict[str, InboxMessage] = {}
        for batch in _batches(new):
            headers_by_uid = await client.fetch_header_fields(batch, ("FROM", "DATE"))
            for uid, (_, headers) in headers_by_uid.items():
                fetched[uid] = InboxMessage(
                    sender=address_of(headers.get("From", "")),
                    date=_parse_date(headers.get("Date", "")),
                )
        for uid in uids:
            message = known.get(uid) or fetched.get(uid)
            if message is not None:
                history.messages[uid] = message.model_copy(
                    update={"seen": uid in seen, "flagged": uid in flagged}
                )
        logger.info(
            "History: %s INBOX messages (%s fetched)", len(history.messages), len(new)
        )

        sent = await client.find_sent_mailbox() if include_sent else None
        if sent:
            await _add_sent(client, sent, history, previous)
        elif include_sent:
            logger.warning("No Sent folder found: 'contacted' features will be 0")
    finally:
        await client.select_mailbox("INBOX")
    return history


async def _add_sent(
    client: "AsyncIMAPClient",
    sent: str,
    history: MailboxHistory,
    previous: MailboxHistory | None,
) -> None:
    """Fill ``contacted`` and ``sent_ids`` from the Sent folder (read-only)."""
    validity = await client.examine_mailbox(sent)
    reuse = (
        previous is not None
        and previous.sent_validity in (None, validity)
        # A history without sent_uids cannot tell which messages it counted
        and (bool(previous.sent_uids) or not previous.contacted)
    )
    contacted: Counter[str] = Counter(previous.contacted if reuse and previous else {})
    sent_ids = list(previous.sent_ids) if reuse and previous else []
    done = set(previous.sent_uids) if reuse and previous else set()
    uids = await client.search_uids("ALL")
    new = [u for u in uids if u not in done]
    fields = ("TO", "CC", "BCC", "MESSAGE-ID")
    for batch in _batches(new):
        for _, headers in (await client.fetch_header_fields(batch, fields)).values():
            # Only present headers: getaddresses() rejects the lot on an empty one
            recipients = [headers[k] for k in ("To", "Cc", "Bcc") if headers.get(k)]
            contacted.update(a.lower() for _, a in getaddresses(recipients) if a)
            sent_ids.extend(message_ids(headers.get("Message-ID", "")))
    history.contacted = dict(contacted)
    history.sent_ids = sent_ids
    history.sent_uids = sorted(done | set(uids), key=int)
    history.sent_validity = validity
    logger.info(
        "History: %s sent messages to %s addresses (%s, %s fetched)",
        len(sent_ids),
        len(contacted),
        sent,
        len(new),
    )


def weeks_between(first: datetime | None, last: datetime | None) -> float:
    """Span in weeks, at least one."""
    if first is None or last is None:
        return 1.0
    return max(1.0, (last - first).total_seconds() / (7 * 86400))


def log_scale(n: float, top: float) -> float:
    """log1p(n) scaled so ``top`` maps to 1 (capped)."""
    return min(1.0, math.log1p(max(n, 0.0)) / math.log1p(top))
