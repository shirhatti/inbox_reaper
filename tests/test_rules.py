"""Tests for rule learning, matching, persistence and pipeline integration."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from inbox_reaper.langgraph_streaming import (
    classify_deterministic,
    run_learning_pipeline,
)
from inbox_reaper.rules import Rule, RuleLearner, RuleMatch, RuleSet
from inbox_reaper.state import Config, Decision, Email, EmailDecision, FilterReason


def make_email(sender: str, uid: str = "1", **kwargs) -> Email:
    return Email(
        uid=uid,
        subject=kwargs.pop("subject", "Hello"),
        sender=sender,
        body=kwargs.pop("body", ""),
        date=datetime(2025, 1, 1),
        **kwargs,
    )


def decided(
    sender: str,
    decision: Decision,
    reason: FilterReason = FilterReason.AI_CLASSIFIED,
) -> EmailDecision:
    return EmailDecision(
        email=make_email(sender), decision=decision, reason=reason, confidence=0.9
    )


def learn(observations, **kwargs) -> RuleSet:
    learner = RuleLearner()
    for obs in observations:
        learner.observe(obs)
    return learner.build(**kwargs)


def rule_keys(rule_set: RuleSet) -> set[tuple[str, str, str]]:
    return {(r.match.value, r.value, r.action.value) for r in rule_set.rules}


class TestRuleLearner:
    def test_learns_consistent_sender(self):
        rules = learn([decided("Deals <deals@shop.example>", Decision.DELETE)] * 3)
        assert ("sender", "deals@shop.example", "delete") in rule_keys(rules)
        assert rules.emails_observed == 3

    def test_requires_min_support(self):
        rules = learn([decided("deals@shop.example", Decision.DELETE)] * 2)
        assert rules.rules == []

    def test_requires_agreement(self):
        obs = [decided("x@shop.example", Decision.DELETE)] * 8
        obs += [decided("x@shop.example", Decision.KEEP)] * 2
        assert learn(obs, min_agreement=0.9).rules == []
        assert learn(obs, min_agreement=0.8).rules

    def test_deterministic_keep_vetoes_delete_rule(self):
        obs = [decided("bank@bank.example", Decision.DELETE)] * 20
        obs.append(decided("bank@bank.example", Decision.KEEP, FilterReason.ATTACHMENT))
        assert learn(obs, min_agreement=0.5).rules == []

    def test_learns_keep_rules(self):
        rules = learn([decided("mom@family.example", Decision.KEEP)] * 3)
        assert ("sender", "mom@family.example", "keep") in rule_keys(rules)

    def test_domain_rule_needs_two_senders(self):
        obs = [decided(f"a{i}@news.example", Decision.DELETE) for i in range(3)]
        rules = learn(obs, min_support=1)
        assert ("domain", "news.example", "delete") in rule_keys(rules)

        single = learn([decided("a@solo.example", Decision.DELETE)] * 3)
        assert not any(r.match == RuleMatch.DOMAIN for r in single.rules)

    def test_no_domain_rules_for_shared_domains(self):
        obs = [decided(f"p{i}@gmail.com", Decision.DELETE) for i in range(5)]
        rules = learn(obs, min_support=1)
        assert not any(r.match == RuleMatch.DOMAIN for r in rules.rules)

    def test_domain_rule_skipped_when_sender_rule_disagrees(self):
        obs = [decided("promo@corp.example", Decision.DELETE)] * 9
        obs += [decided("hr@corp.example", Decision.KEEP)] * 3
        rules = learn(obs, min_agreement=0.7)
        assert ("sender", "hr@corp.example", "keep") in rule_keys(rules)
        assert not any(r.match == RuleMatch.DOMAIN for r in rules.rules)

    def test_ignores_uncertain_decisions(self):
        rules = learn([decided("x@shop.example", Decision.UNCERTAIN)] * 5)
        assert rules.emails_observed == 0


class TestRuleSet:
    @pytest.fixture
    def rule_set(self) -> RuleSet:
        return RuleSet(
            rules=[
                Rule(
                    match=RuleMatch.SENDER,
                    value="vip@shop.example",
                    action=Decision.KEEP,
                ),
                Rule(
                    match=RuleMatch.DOMAIN, value="shop.example", action=Decision.DELETE
                ),
            ]
        )

    def test_sender_beats_domain(self, rule_set):
        assert rule_set.match(make_email("VIP <VIP@shop.example>")).action == "keep"

    def test_domain_matches_subdomains(self, rule_set):
        assert rule_set.match(make_email("x@mail.shop.example")).action == "delete"

    def test_no_match_on_lookalike_or_tld(self, rule_set):
        assert rule_set.match(make_email("x@evilshop.example")) is None
        assert rule_set.match(make_email("x@other.example")) is None

    def test_decide_uses_rule_reason(self, rule_set):
        decision = rule_set.decide(make_email("x@shop.example"))
        assert decision.decision == Decision.DELETE
        assert decision.reason == FilterReason.RULE

    def test_round_trip(self, rule_set, tmp_path):
        path = tmp_path / "rules.json"
        rule_set.save(path)
        loaded = RuleSet.load(path)
        assert loaded.rules == rule_set.rules
        assert loaded.match(make_email("x@shop.example")).action == "delete"

    def test_rejects_unknown_version(self, tmp_path):
        path = tmp_path / "rules.json"
        path.write_text('{"version": 99, "rules": []}')
        with pytest.raises(ValueError):
            RuleSet.load(path)


class TestRuleApplication:
    def test_hard_keep_filters_override_rules(self):
        rules = RuleSet(
            rules=[Rule(match=RuleMatch.DOMAIN, value="bank.example", action="delete")]
        )
        email = make_email("a@bank.example", attachments=["statement.pdf"])
        decision = classify_deterministic(email, Config(), rules)
        assert decision.decision == Decision.KEEP
        assert decision.reason == FilterReason.ATTACHMENT

    def test_rules_apply_before_heuristics(self):
        rules = RuleSet(
            rules=[
                Rule(match=RuleMatch.SENDER, value="news@shop.example", action="keep")
            ]
        )
        # The marketing heuristic alone would delete this email
        email = make_email("news@shop.example", body="Unsubscribe here")
        assert classify_deterministic(email, Config()).decision == Decision.DELETE
        decision = classify_deterministic(email, Config(), rules)
        assert decision.decision == Decision.KEEP
        assert decision.reason == FilterReason.RULE


class TestLearningPipeline:
    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    def test_learning_never_mutates_mailbox(self, mock_create_client):
        messages = {
            str(i): make_email(
                f"deals{i % 2}@e.shop.example", uid=str(i), body="Unsubscribe now"
            )
            for i in range(1, 7)
        }
        client = AsyncMock()
        client.search_uids.return_value = list(messages)
        client.fetch_messages.side_effect = lambda uids: {u: messages[u] for u in uids}
        mock_create_client.return_value = client

        rule_set = asyncio.run(
            run_learning_pipeline(
                Config(email="me@example.com", dry_run=False), use_ai=False
            )
        )

        client.delete_emails.assert_not_called()
        assert rule_set.emails_observed == 6
        assert ("domain", "e.shop.example", "delete") in rule_keys(rule_set)
        assert ("sender", "deals0@e.shop.example", "delete") in rule_keys(rule_set)

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    def test_no_ai_skips_undecided_emails(self, mock_create_client):
        client = AsyncMock()
        client.search_uids.return_value = ["1"]
        client.fetch_messages.return_value = {"1": make_email("x@corp.example")}
        mock_create_client.return_value = client

        with patch("inbox_reaper.langgraph_streaming.classify_with_ai") as mock_ai:
            rule_set = asyncio.run(
                run_learning_pipeline(Config(email="me@example.com"), use_ai=False)
            )
            mock_ai.assert_not_called()
        assert rule_set.emails_observed == 0


class TestProcessWithRules:
    @patch("inbox_reaper.langgraph_streaming.classify_with_ai")
    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    def test_rule_drives_delete(self, mock_create_client, mock_ai, tmp_path):
        from inbox_reaper.langgraph_streaming import run_streaming_pipeline

        path = tmp_path / "rules.json"
        RuleSet(
            rules=[
                Rule(match=RuleMatch.SENDER, value="bot@corp.example", action="delete")
            ]
        ).save(path)
        client = AsyncMock()
        client.search_uids.return_value = ["1"]
        client.fetch_messages.return_value = {"1": make_email("Bot <bot@corp.example>")}
        mock_create_client.return_value = client

        config = Config(email="me@example.com", dry_run=False, rules_path=str(path))
        asyncio.run(run_streaming_pipeline(config))

        mock_ai.assert_not_called()
        client.delete_emails.assert_awaited_once_with(["1"])

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    def test_bad_rules_file_fails_before_connecting(self, mock_create_client, tmp_path):
        from inbox_reaper.langgraph_streaming import run_streaming_pipeline

        path = tmp_path / "rules.json"
        path.write_text("not json")
        config = Config(email="me@example.com", rules_path=str(path))
        with pytest.raises(ValueError):
            asyncio.run(run_streaming_pipeline(config))
        mock_create_client.assert_not_called()
