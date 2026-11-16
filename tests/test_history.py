"""Tests for mailbox history: IMAP parsing, building, features and guards."""

import asyncio
import email
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from inbox_reaper.combiner import TrainingExample, train
from inbox_reaper.features import FeatureContext, history_features
from inbox_reaper.history import InboxMessage, MailboxHistory, build_history
from inbox_reaper.imap_client import parse_fetch_flags, parse_sent_mailbox
from inbox_reaper.langgraph_streaming import classify_email
from inbox_reaper.state import Config, Decision, Email

NOW = datetime(2026, 11, 15, tzinfo=UTC)
CONFIG = Config(email="me@example.com")


def make_email(uid: str, sender: str, **headers: str) -> Email:
    return Email(
        uid=uid,
        subject="Hello",
        sender=sender,
        body="",
        date=NOW - timedelta(days=1),
        headers=headers,
    )


def inbox(*rows: tuple[str, str, bool, bool, int]) -> dict[str, InboxMessage]:
    """(uid, sender, seen, flagged, days ago) rows -> history messages."""
    return {
        uid: InboxMessage(
            sender=sender, seen=seen, flagged=flagged, date=NOW - timedelta(days=d)
        )
        for uid, sender, seen, flagged, d in rows
    }


HISTORY = MailboxHistory(
    messages=inbox(
        ("1", "news@shop.example", False, False, 1),
        ("2", "news@shop.example", False, False, 8),
        ("3", "news@shop.example", False, False, 15),
        ("4", "news@shop.example", True, False, 22),
        ("5", "alex@corp.example", True, True, 3),
        ("6", "alex@corp.example", True, False, 30),
        ("7", "pat@gmail.com", True, False, 2),
    ),
    contacted={"alex@corp.example": 4, "friend@gmail.com": 9},
    sent_ids=["<mine-1@example.com>"],
)


def features(email: Email, history: MailboxHistory | None = HISTORY):
    return history_features(email, FeatureContext(CONFIG, now=NOW, history=history))


class TestParsing:
    def test_flags_before_or_after_the_literal(self):
        lines = [
            b"1 FETCH (UID 5 FLAGS (\\Seen \\Flagged) BODY[HEADER.FIELDS (FROM)] {9}",
            bytearray(b"From: a\r\n"),
            b")",
            b"2 FETCH (UID 7 BODY[HEADER.FIELDS (FROM)] {9}",
            bytearray(b"From: b\r\n"),
            b" FLAGS ())",
        ]
        assert parse_fetch_flags(lines) == {
            "5": frozenset({"\\Seen", "\\Flagged"}),
            "7": frozenset(),
        }

    def test_sent_mailbox_special_use_then_names(self):
        marked = [b'(\\HasNoChildren) "/" Inbox', b'(\\Sent) "/" "Sent Items"']
        assert parse_sent_mailbox(marked) == "Sent Items"
        assert parse_sent_mailbox([b'(\\HasNoChildren) "/" "Sent"']) == "Sent"
        assert parse_sent_mailbox([b'(\\HasNoChildren) "/" Archive']) is None


class TestBuildHistory:
    def test_reads_inbox_and_sent_read_only(self):
        client = AsyncMock()
        # INBOX: ALL, SEEN, FLAGGED; then Sent: ALL
        client.search_uids.side_effect = [["10", "11"], ["10"], [], ["90"]]
        client.examine_mailbox.return_value = "7"
        client.find_sent_mailbox.return_value = "Sent Items"

        def header(text: str):
            return email.message_from_string(text)

        client.fetch_header_fields.side_effect = [
            {
                "10": (
                    frozenset({"\\Seen"}),
                    header("From: A <A@x.example>\nDate: Mon, 06 Jan 2025 10:00 +0000"),
                ),
                "11": (frozenset(), header("From: b@y.example\n")),
            },
            {
                "90": (
                    frozenset(),
                    header("To: A <a@x.example>, c@z.example\nMessage-ID: <s1@me>\n"),
                )
            },
        ]

        history = asyncio.run(build_history(client))

        assert [c.args[0] for c in client.examine_mailbox.call_args_list] == [
            "INBOX",
            "Sent Items",
        ]
        client.select_mailbox.assert_awaited_once_with("INBOX")
        assert history.messages["10"].sender == "a@x.example"
        assert history.messages["10"].seen and not history.messages["11"].seen
        assert history.contacted == {"a@x.example": 1, "c@z.example": 1}
        assert history.sent_ids == ["<s1@me>"]
        assert (history.inbox_validity, history.sent_uids) == ("7", ["90"])
        # Never a command that could change the mailbox
        assert not client.delete_emails.called
        assert not client.fetch_messages.called

    def update_client(self, validity: str = "7") -> AsyncMock:
        client = AsyncMock()
        # INBOX now: 10 (still there, now read), 12 (new); 11 is gone
        client.search_uids.side_effect = [["10", "12"], ["10"], ["12"], ["90", "91"]]
        client.examine_mailbox.return_value = validity
        client.find_sent_mailbox.return_value = "Sent"
        client.fetch_header_fields.side_effect = [
            {
                "12": (
                    frozenset(),
                    email.message_from_string("From: new@x.example\n"),
                )
            },
            {"91": (frozenset(), email.message_from_string("To: d@z.example\n"))},
        ]
        return client

    def previous(self) -> MailboxHistory:
        return MailboxHistory(
            messages=inbox(
                ("10", "a@x.example", False, False, 3),
                ("11", "b@y.example", False, False, 2),
            ),
            contacted={"a@x.example": 1},
            sent_ids=["<s1@me>"],
            sent_uids=["90"],
            inbox_validity="7",
            sent_validity="7",
        )

    def test_update_fetches_only_new_messages_and_refreshes_flags(self):
        client = self.update_client()
        history = asyncio.run(build_history(client, previous=self.previous()))
        fetched = [c.args[0] for c in client.fetch_header_fields.call_args_list]
        assert fetched == [["12"], ["91"]]
        assert sorted(history.messages) == ["10", "12"]
        assert history.messages["10"].seen  # read since the last scan
        assert history.messages["12"].flagged
        assert history.contacted == {"a@x.example": 1, "d@z.example": 1}
        assert history.sent_uids == ["90", "91"]

    def test_changed_uidvalidity_rebuilds(self):
        client = self.update_client(validity="8")
        client.search_uids.side_effect = [["10"], [], [], ["90"]]
        client.fetch_header_fields.side_effect = [
            {"10": (frozenset(), email.message_from_string("From: z@q.example\n"))},
            {"90": (frozenset(), email.message_from_string("To: e@z.example\n"))},
        ]
        history = asyncio.run(build_history(client, previous=self.previous()))
        # UID 10 was renumbered: its old sender is not reused
        assert history.messages["10"].sender == "z@q.example"
        assert history.contacted == {"e@z.example": 1}

    def test_round_trip(self, tmp_path):
        path = tmp_path / "history.json"
        HISTORY.save(path)
        loaded = MailboxHistory.load(path)
        assert loaded.by_sender["news@shop.example"].count == 4


class TestHistoryFeatures:
    def test_without_history_only_marks_unavailable(self):
        assert features(make_email("1", "news@shop.example"), None) == {
            "available": 0.0
        }

    def test_never_read_sender_excluding_the_email_itself(self):
        f = features(make_email("1", "News <news@shop.example>"))
        assert f["available"] == 1.0 and f["seen"] == 0.0
        # Three other emails from this sender, one of them read
        assert f["sender_read_rate"] == pytest.approx(1 / 3)
        assert f["sender_never_read"] == 0.0
        assert f["sender_new"] == 0.0
        assert f["contacted_sender"] == 0.0

    def test_own_read_state_is_not_counted_as_history(self):
        # Alex's other email was read; this one's own flags do not leak in
        f = features(make_email("5", "alex@corp.example"))
        assert (f["seen"], f["flagged"]) == (1.0, 1.0)
        assert f["sender_read_rate"] == 1.0 and f["sender_flagged"] == 0.0
        assert f["contacted_sender"] == 1.0 and f["contacted_domain"] == 1.0

    def test_new_sender_and_free_mail_domains(self):
        f = features(make_email("99", "stranger@gmail.com"))
        assert f["sender_new"] == 1.0 and f["seen"] == 0.0
        # You wrote to friend@gmail.com, which says nothing about this sender
        assert f["contacted_domain"] == 0.0

    def test_my_thread(self):
        reply = make_email(
            "99", "x@corp.example", references="<a@b> <mine-1@example.com>"
        )
        assert features(reply)["my_thread"] == 1.0
        assert features(make_email("98", "x@corp.example"))["my_thread"] == 0.0

    def test_stale_uid_is_ignored(self):
        # UID 1 now belongs to someone else: its flags must not be used
        f = features(make_email("5", "other@else.example"))
        assert f["seen"] == 0.0 and f["sender_new"] == 1.0


class TestNeedsHistory:
    def examples(self, available: float) -> list[TrainingExample]:
        return [
            TrainingExample(
                features={"history.available": available, "x": float(i % 2)},
                label=Decision.DELETE if i % 2 else Decision.KEEP,
            )
            for i in range(20)
        ]

    def test_combiner_remembers_it_was_trained_with_history(self):
        assert train(self.examples(1.0)).needs_history
        assert not train(self.examples(0.0)).needs_history

    def test_refuses_to_score_without_history(self):
        combiner = train(self.examples(1.0))
        ai = AsyncMock()
        with pytest.raises(ValueError, match="history"):
            asyncio.run(
                classify_email(
                    make_email("1", "a@b.example"), CONFIG, None, ai, combiner
                )
            )
