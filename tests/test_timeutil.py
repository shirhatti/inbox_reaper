"""Tests for timezone-aware timestamps and migration of legacy naive values."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from inbox_reaper.imap_client import _parse_date
from inbox_reaper.oauth_flow import refresh_if_expiring, token_needs_refresh
from inbox_reaper.timeutil import expiry_after, parse_timestamp, to_utc, utcnow


class TestParseTimestamp:
    def test_aware_string_kept_as_instant(self):
        parsed = parse_timestamp("2025-06-01T12:00:00+02:00")
        assert parsed == datetime(2025, 6, 1, 10, 0, tzinfo=UTC)
        assert parsed.tzinfo is UTC

    def test_legacy_naive_string_is_local_time(self):
        legacy = "2025-06-01T12:00:00"
        expected = datetime.fromisoformat(legacy).astimezone().astimezone(UTC)
        assert parse_timestamp(legacy) == expected

    def test_unix_timestamp(self):
        assert parse_timestamp(0) == datetime(1970, 1, 1, tzinfo=UTC)

    @pytest.mark.parametrize("bad", [None, True, ["2025"]])
    def test_rejects_non_timestamps(self, bad):
        with pytest.raises(TypeError):
            parse_timestamp(bad)

    def test_rejects_garbage_string(self):
        with pytest.raises(ValueError):
            parse_timestamp("next tuesday")


def test_to_utc_naive_as_utc():
    assert to_utc(datetime(2025, 1, 1), naive_is_local=False) == datetime(
        2025, 1, 1, tzinfo=UTC
    )


def test_expiry_after_is_aware_utc():
    expiry = datetime.fromisoformat(expiry_after(60))
    assert expiry.utcoffset() == timedelta(0)
    assert timedelta(seconds=55) < expiry - utcnow() <= timedelta(seconds=60)


class TestEmailDates:
    def test_offset_converted_to_utc(self):
        parsed = _parse_date("Mon, 06 Jan 2025 10:00:00 +0200")
        assert parsed == datetime(2025, 1, 6, 8, 0, tzinfo=UTC)

    def test_unknown_zone_means_utc(self):
        # RFC 5322 "-0000": UTC with no local zone info; stdlib returns naive
        parsed = _parse_date("Mon, 06 Jan 2025 10:00:00 -0000")
        assert parsed == datetime(2025, 1, 6, 10, 0, tzinfo=UTC)

    def test_unparseable_falls_back_to_now(self):
        assert _parse_date("garbage").tzinfo is not None


def legacy(dt_local_offset: timedelta) -> str:
    """A naive local-time ISO string, as older versions stored expires_at."""
    return (datetime.now() + dt_local_offset).isoformat()


class TestTokenNeedsRefresh:
    def test_legacy_naive_expiry_in_future(self):
        assert not token_needs_refresh({"expires_at": legacy(timedelta(hours=1))})

    def test_legacy_naive_expiry_in_past(self):
        assert token_needs_refresh({"expires_at": legacy(-timedelta(hours=1))})

    def test_within_margin_counts_as_expiring(self):
        assert token_needs_refresh({"expires_at": expiry_after(60)})

    def test_aware_expiry(self):
        assert not token_needs_refresh({"expires_at": expiry_after(3600)})

    @pytest.mark.parametrize("creds", [{}, {"expires_at": None}, {"expires_at": "?"}])
    def test_missing_or_invalid_counts_as_expiring(self, creds):
        assert token_needs_refresh(creds)


@patch("inbox_reaper.oauth_flow.store_credentials")
@patch("inbox_reaper.oauth_flow.refresh_access_token")
def test_refresh_migrates_legacy_expiry_to_aware(mock_refresh, mock_store):
    mock_refresh.return_value = {"access_token": "new", "expires_in": 3600}
    creds = {
        "access_token": "old",
        "refresh_token": "r",
        "provider": "gmail",
        "expires_at": legacy(-timedelta(minutes=1)),
    }

    assert refresh_if_expiring("me@example.com", creds) is True

    assert creds["access_token"] == "new"
    assert datetime.fromisoformat(creds["expires_at"]).utcoffset() == timedelta(0)
    mock_store.assert_called_once_with("me@example.com", creds)


@patch("inbox_reaper.oauth_flow.refresh_access_token")
def test_no_refresh_without_refresh_token(mock_refresh):
    assert refresh_if_expiring("me@example.com", {"expires_at": None}) is False
    mock_refresh.assert_not_called()
