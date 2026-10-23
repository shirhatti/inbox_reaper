"""Tests for IMAP response parsing and message extraction (no network)."""

import email
from email.message import EmailMessage
from unittest.mock import AsyncMock

import pytest

from inbox_reaper.imap_client import (
    MAX_BODY_CHARS,
    AsyncIMAPClient,
    IMAPConnectionError,
    parse_fetch_literals,
)


def make_client() -> AsyncIMAPClient:
    return AsyncIMAPClient("me@example.com", "token", provider="gmail")


def raw_message(**headers: str) -> bytes:
    msg = EmailMessage()
    msg["From"] = headers.get("From", "Sender <sender@example.com>")
    msg["Subject"] = headers.get("Subject", "Hello")
    msg["Date"] = "Mon, 06 Jan 2025 10:00:00 +0000"
    if "List-Unsubscribe" in headers:
        msg["List-Unsubscribe"] = headers["List-Unsubscribe"]
    msg.set_content(headers.get("Body", "Plain body"))
    return msg.as_bytes()


class TestParseFetchLiterals:
    def test_uid_before_literal(self):
        lines = [
            b"1 FETCH (UID 101 BODY[] {5}",
            bytearray(b"first"),
            b")",
            b"2 FETCH (UID 102 BODY[] {6}",
            bytearray(b"second"),
            b")",
            b"Success",
        ]
        assert parse_fetch_literals(lines) == {"101": b"first", "102": b"second"}

    def test_uid_after_literal(self):
        lines = [
            b"1 FETCH (BODY[] {5}",
            bytearray(b"first"),
            b" UID 101)",
            b"2 FETCH (BODY[] {6}",
            bytearray(b"second"),
            b" UID 102 FLAGS (\\Seen))",
        ]
        assert parse_fetch_literals(lines) == {"101": b"first", "102": b"second"}

    def test_short_literals_are_kept(self):
        # The old parser dropped any literal of 100 bytes or fewer
        lines = [b"1 FETCH (UID 7 BODY[HEADER] {9}", bytearray(b"Subject:x"), b")"]
        assert parse_fetch_literals(lines) == {"7": b"Subject:x"}

    def test_ignores_status_lines(self):
        assert parse_fetch_literals([b"FETCH completed"]) == {}


class TestMessageParsing:
    def test_parse_message_fields(self):
        msg = make_client()._parse_message(
            "5",
            raw_message(
                Subject="=?utf-8?q?Caf=C3=A9?=",
                **{"List-Unsubscribe": "<mailto:u@example.com>"},
            ),
        )
        assert msg.uid == "5"
        assert msg.subject == "Café"
        assert msg.sender == "Sender <sender@example.com>"
        assert msg.body.strip() == "Plain body"
        assert msg.has_list_unsubscribe is True

    def test_list_unsubscribe_absent(self):
        msg = make_client()._parse_message("5", raw_message())
        assert msg.has_list_unsubscribe is False

    def test_html_body_is_stripped_and_keeps_footer(self):
        msg = EmailMessage()
        msg["From"] = "shop@example.com"
        msg.set_content("x")
        style = "<style>" + ("p{color:red}" * 500) + "</style>"
        filler = "<p>" + ("Lots of content. " * 1000) + "</p>"
        msg.add_alternative(
            f"<html><head>{style}</head><body>{filler}"
            "<a href='#'>Unsubscribe</a></body></html>",
            subtype="html",
        )
        # Drop the text/plain part so the HTML fallback is used
        msg.get_payload().pop(0)

        body = make_client()._extract_body(email.message_from_bytes(msg.as_bytes()))

        assert "<" not in body
        assert "color:red" not in body
        assert body.endswith("Unsubscribe")
        assert len(body) <= MAX_BODY_CHARS

    def test_respects_declared_charset(self):
        msg = EmailMessage()
        msg.set_content("Grüße", charset="iso-8859-1", cte="8bit")
        body = make_client()._extract_body(email.message_from_bytes(msg.as_bytes()))
        assert body.strip() == "Grüße"


class TestFetchMessages:
    async def _client_with_response(self, result: str, lines: list) -> AsyncIMAPClient:
        client = make_client()
        client.client = AsyncMock()
        client.client.uid.return_value = AsyncMock(result=result, lines=lines)
        return client

    @pytest.mark.parametrize("uids", [["1", "2"]])
    def test_batches_into_single_fetch(self, uids):
        import asyncio

        lines = []
        for uid in uids:
            lines += [f"{uid} FETCH (UID {uid} BODY[] {{1}}".encode()]
            lines += [bytearray(raw_message(Subject=f"S{uid}")), b")"]

        async def run():
            client = await self._client_with_response("OK", lines)
            messages = await client.fetch_messages(uids)
            return client, messages

        client, messages = asyncio.run(run())

        assert client.client.uid.await_count == 1
        args = client.client.uid.await_args.args
        assert args[:2] == ("fetch", "1,2")
        assert "BODY.PEEK[]" in args[2]
        assert {m.subject for m in messages.values()} == {"S1", "S2"}

    def test_timeout_raises_connection_error(self):
        import asyncio

        async def run():
            client = make_client()
            client.timeout = 0.01
            client.client = AsyncMock()

            async def hang(*args):
                await asyncio.sleep(1)

            client.client.uid.side_effect = hang
            await client.fetch_messages(["1"])

        with pytest.raises(IMAPConnectionError):
            asyncio.run(run())
