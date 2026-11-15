"""Tests for the feature processors, time sensitivity and text helpers."""

from datetime import UTC, datetime, timedelta

import pytest

from inbox_reaper.features import (
    FeatureContext,
    Lifetime,
    check_expiry,
    describe_features,
    email_lifetime,
    extract_features,
    header_features,
    link_features,
    rule_features,
    sender_features,
    temporal_features,
)
from inbox_reaper.rules import Rule, RuleMatch, RuleSet
from inbox_reaper.state import Config, Decision, Email, FilterReason
from inbox_reaper.text_utils import (
    compact_urls,
    html_structure,
    real_url,
    strip_html,
    truncate_symmetric,
    url_domain,
)

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
CTX = FeatureContext(config=Config(email="me@example.com"), now=NOW)


def make_email(
    subject: str = "Hello",
    sender: str = "Pat <pat@shop.example>",
    body: str = "",
    age: timedelta = timedelta(days=1),
    **fields,
) -> Email:
    return Email(
        uid="1", subject=subject, sender=sender, body=body, date=NOW - age, **fields
    )


SAFELINK = (
    "https://nam04.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.chase.com"
    "%2Fstatement&data=05%7C01&reserved=0"
)


class TestTextHelpers:
    def test_real_url_unwraps_safe_links(self):
        assert real_url(SAFELINK) == "https://www.chase.com/statement"
        assert real_url("https://example.com/a") == "https://example.com/a"

    def test_url_domain(self):
        assert url_domain(SAFELINK) == "chase.com"
        assert url_domain("http://user@Mail.Example.com:8080/x") == "mail.example.com"

    def test_compact_urls_keeps_only_domain(self):
        text = f"See {SAFELINK} or https://click.e.shop.example/t?id=123456789 now"
        assert compact_urls(text) == (
            "See [link: chase.com] or [link: click.e.shop.example] now"
        )

    def test_html_structure_links_images_and_pixels(self):
        markup = (
            f'<a href="{SAFELINK}">Statement</a><a href="mailto:x@y.z">mail</a>'
            '<img src="logo.png" width="120"><img src="p.gif" width="1" height="1">'
            '<img src="h.gif" style="display: none">'
        )
        s = html_structure(markup)
        assert s.links == ["https://www.chase.com/statement"]
        assert (s.images, s.tracking_pixels) == (3, 2)

    def test_strip_html_drops_invisible_spacers(self):
        assert strip_html("Sale‌ ‌͏­today") == "Sale today"

    def test_truncate_compacts_urls_first(self):
        body = "Hi " + "https://t.example/" + "x" * 5000
        assert truncate_symmetric(body, 100) == "Hi [link: t.example]"


class TestHeaderFeatures:
    def test_bulk_mail_headers(self):
        email = make_email(
            has_list_unsubscribe=True,
            headers={
                "list-id": "<news.shop.example>",
                "precedence": "bulk",
                "list-unsubscribe-post": "List-Unsubscribe=One-Click",
                "x-sg-eid": "abc",
                "dkim-signature": "v=1; a=rsa-sha256; d=mail.shop.example; s=s1",
                "authentication-results": "spf=pass; dkim=pass; dmarc=pass",
                "reply-to": "replies@esp.example",
                "x-microsoft-antispam": "BCL:7;",
            },
        )
        f = header_features(email, CTX)
        for flag in (
            "list_id",
            "list_unsubscribe",
            "one_click_unsubscribe",
            "precedence_bulk",
            "esp",
            "dkim",
            "dkim_aligned",
            "spf_pass",
            "dmarc_pass",
            "reply_to_elsewhere",
        ):
            assert f[flag] == 1.0, flag
        assert f["bulk_complaint_level"] == pytest.approx(7 / 9)
        assert f["reply_thread"] == 0.0

    def test_conversation_and_misaligned_dkim(self):
        email = make_email(
            subject="RE: Q4 plan",
            headers={"in-reply-to": "<a@b>", "dkim-signature": "d=esp.example;"},
        )
        f = header_features(email, CTX)
        assert (f["reply_thread"], f["subject_reply"]) == (1.0, 1.0)
        assert (f["dkim"], f["dkim_aligned"]) == (1.0, 0.0)

    def test_github_reason(self):
        mention = make_email(headers={"x-github-reason": "mention"})
        subscribed = make_email(headers={"x-github-reason": "subscribed"})
        assert header_features(mention, CTX)["github_direct"] == 1.0
        f = header_features(subscribed, CTX)
        assert (f["github"], f["github_direct"]) == (1.0, 0.0)


class TestOtherProcessors:
    def test_link_features(self):
        email = make_email(
            links=[
                "https://click.e.shop.example/abc",
                "https://shop.example/item?utm_source=mail",
                "https://other.example/",
                "https://shop.example/about",
            ],
            image_count=6,
            tracking_pixels=1,
        )
        f = link_features(email, CTX)
        assert f["tracking_share"] == 0.5
        assert f["offsite_share"] == 0.25
        assert f["tracking_pixel"] == 1.0

    def test_sender_features(self):
        assert sender_features(make_email(sender="Al <al@gmail.com>"), CTX) == {
            "freemail": 1.0,
            "automated_local_part": 0.0,
            "subdomain_sender": 0.0,
            "has_display_name": 1.0,
        }
        f = sender_features(make_email(sender="no-reply@mail.bank.example"), CTX)
        assert (f["automated_local_part"], f["subdomain_sender"]) == (1.0, 1.0)

    def test_rule_features(self):
        rules = RuleSet(
            rules=[Rule(match=RuleMatch.DOMAIN, value="shop.example", action="delete")]
        )
        ctx = FeatureContext(config=CTX.config, rules=rules, now=NOW)
        assert rule_features(make_email(), ctx) == {"keep": 0.0, "delete": 1.0}
        assert rule_features(make_email(), CTX) == {"keep": 0.0, "delete": 0.0}

    def test_extract_features_prefixes_and_describe(self):
        email = make_email(headers={"precedence": "bulk", "list-id": "<x>"})
        features = extract_features(email, CTX)
        assert features["header.precedence_bulk"] == 1.0
        assert "filter.marketing_delete" in features
        assert all("." in name for name in features)
        summary = describe_features(features)
        assert "bulk mail" in summary and "mailing list" in summary


class TestTimeSensitivity:
    @pytest.mark.parametrize(
        ("subject", "body", "expected"),
        [
            ("Your verification code is 482915", "", Lifetime.MINUTES),
            (
                "Sign in to Contoso",
                "Use this one-time passcode: 1234",
                Lifetime.MINUTES,
            ),
            (
                "Verify your email",
                "Click to verify your email address",
                Lifetime.MINUTES,
            ),
            ("Gate change for AS 3301", "", Lifetime.MINUTES),
            (
                "Your Daily Digest for Mon, Jun 13",
                "Informed Delivery",
                Lifetime.MINUTES,
            ),
            ("Out for delivery today", "", Lifetime.DAYS),
            ("New sign-in to your account", "We noticed a new sign-in", Lifetime.DAYS),
            ("Your reservation is confirmed", "", Lifetime.DAYS),
            # A booking confirmation code is a record, not a one-time code
            ("Flight confirmation", "Confirmation code: K8P2QZ", Lifetime.FOREVER),
            ("Your receipt", "Order total $42.10", Lifetime.FOREVER),
        ],
    )
    def test_email_lifetime(self, subject, body, expected):
        assert email_lifetime(make_email(subject, body=body)) == expected

    @pytest.mark.parametrize(
        ("age", "expected"),
        [
            (timedelta(minutes=10), Decision.KEEP),
            (timedelta(minutes=45), None),  # between fresh and expired: undecided
            (timedelta(hours=2), Decision.DELETE),
        ],
    )
    def test_one_time_code_expiry(self, age, expected):
        email = make_email("Your verification code is 482915", age=age)
        decision = check_expiry(email, CTX.config, now=NOW)
        assert (decision.decision if decision else None) == expected
        if decision:
            assert decision.reason in {
                FilterReason.TIME_SENSITIVE,
                FilterReason.EXPIRED,
            }

    def test_day_scale_mail_is_never_deleted_by_the_rule(self):
        # Only the combiner sees day-scale expiry (as a feature): its patterns
        # are too broad to delete on their own
        old = make_email("Out for delivery today", age=timedelta(days=400))
        assert check_expiry(old, CTX.config, now=NOW) is None
        f = temporal_features(old, CTX)
        assert (f["lifetime_days"], f["expired"]) == (1.0, 1.0)

    def test_regular_mail_is_never_expired(self):
        old = make_email("Your receipt", age=timedelta(days=3000))
        assert check_expiry(old, CTX.config, now=NOW) is None

    def test_temporal_features(self):
        f = temporal_features(
            make_email("Your code is 1234", age=timedelta(hours=3)), CTX
        )
        assert (f["lifetime_minutes"], f["expired"], f["fresh"]) == (1.0, 1.0, 0.0)
        assert 0 < f["log_age_days"] < 1
