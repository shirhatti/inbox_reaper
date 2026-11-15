"""Tests for sampling files, interactive labelling, comparison and training."""

import random
from datetime import UTC, datetime, timedelta

from click.testing import CliRunner

from inbox_reaper.cli import cli
from inbox_reaper.combiner import Combiner
from inbox_reaper.features import Lifetime
from inbox_reaper.labeling import (
    Label,
    Loss,
    Sample,
    append_jsonl,
    compare,
    disagreements,
    evaluate,
    format_evaluation,
    load_labels,
    review_order,
    training_examples,
)
from inbox_reaper.rules import Rule, RuleMatch, RuleSet
from inbox_reaper.state import Config, Decision, Email

NOW = datetime(2026, 11, 15, tzinfo=UTC)


def make_sample(
    uid: str, subject: str = "Hi", sender: str = "a@b.example", **kw
) -> Sample:
    email = Email(
        uid=uid,
        subject=subject,
        sender=sender,
        body=kw.pop("body", "Hello"),
        date=NOW - timedelta(days=2),
        **kw,
    )
    return Sample(uid=uid, sampled_at=NOW, email=email)


def lab(uid: str, action: str, labeler: str = "x", **kw) -> Label:
    return Label(uid=uid, labeler=labeler, action=Decision(action), **kw)


def write_samples(path, samples):
    for s in samples:
        append_jsonl(path, s)


class TestLabelCommand:
    def test_asks_every_dimension_and_resumes(self, tmp_path):
        samples_path, labels_path = tmp_path / "s.jsonl", tmp_path / "l.jsonl"
        write_samples(
            samples_path, [make_sample("1"), make_sample("2"), make_sample("3")]
        )
        runner = CliRunner()
        # email 1: keep / forever / category 3 (receipt) / bad loss / a note
        # email 2: skip; email 3: quit
        answers = "k\nf\n3\nb\nmy receipt\ns\nq\n"
        result = runner.invoke(
            cli, ["label", str(samples_path), "-o", str(labels_path)], input=answers
        )
        assert result.exit_code == 0, result.output
        labels = load_labels(labels_path)
        assert list(labels) == ["1"]
        first = labels["1"]
        assert (
            first.action,
            first.lifetime,
            first.category,
            first.loss,
            first.note,
        ) == (
            Decision.KEEP,
            Lifetime.FOREVER,
            "receipt",
            Loss.BAD,
            "my receipt",
        )
        assert first.labeler == "human"

        # Resuming skips already-labelled emails: next prompt is for email 2
        answers = "d\nm\n14\nn\n\nq\n"
        result = runner.invoke(
            cli, ["label", str(samples_path), "-o", str(labels_path)], input=answers
        )
        labels = load_labels(labels_path)
        assert labels["2"].action == Decision.DELETE
        assert labels["2"].lifetime == Lifetime.MINUTES
        assert labels["2"].category == "survey"
        assert "3" not in labels


class TestReviewOrder:
    def test_disagreements_first(self):
        samples = [make_sample(str(i)) for i in range(6)]
        a = {"1": lab("1", "keep"), "2": lab("2", "keep"), "4": lab("4", "keep")}
        b = {
            "1": lab("1", "delete"),  # action disagreement -> first
            "2": lab("2", "keep", lifetime=Lifetime.DAYS),  # lifetime only -> second
            "4": lab("4", "keep"),
        }
        order = [s.uid for s in review_order(samples, [a, b])]
        assert order[0] == "1" and order[1] == "2"
        assert sorted(order) == [str(i) for i in range(6)]


class TestCompare:
    def test_agreement_and_disagreements(self):
        human = {"1": lab("1", "keep"), "2": lab("2", "delete"), "3": lab("3", "keep")}
        model = {
            "1": lab("1", "keep"),
            "2": lab("2", "keep"),
            "3": lab("3", "keep", category="promo"),
        }
        lines = compare({"human": human, "model": model})
        assert lines[0] == "human vs model (3 emails in common)"
        assert "action    agreement   67%" in lines[1]
        assert "category  agreement   67%" in lines[3]
        assert "human delete / model keep: 1" in lines[4]
        samples = {"2": make_sample("2", subject="Big sale")}
        rows = list(disagreements({"human": human, "model": model}, samples))
        assert len(rows) == 1 and "Big sale" in rows[0]


class TestTraining:
    def test_examples_use_first_label_set_and_loss_weights(self):
        samples = [make_sample("1"), make_sample("2"), make_sample("3")]
        human = {"1": lab("1", "keep", loss=Loss.BAD)}
        model = {"1": lab("1", "delete"), "2": lab("2", "delete")}
        examples = training_examples(samples, [human, model], Config(email="me@x"))
        assert [e.label for e in examples] == [Decision.KEEP, Decision.DELETE]
        assert examples[0].weight == 10.0 and examples[1].weight == 1.0
        assert "header.list_id" in examples[0].features

    def test_expired_lifetime_turns_keep_into_delete(self):
        sample = make_sample("1")  # sampled two days after it arrived
        examples = training_examples(
            [sample],
            [{"1": lab("1", "keep", lifetime=Lifetime.MINUTES, loss=Loss.BAD)}],
            Config(email="me@x"),
        )
        assert examples[0].label == Decision.DELETE and examples[0].weight == 1.0
        still_useful = training_examples(
            [sample],
            [{"1": lab("1", "keep", lifetime=Lifetime.DAYS)}],
            Config(email="me@x"),
        )
        assert still_useful[0].label == Decision.KEEP

    def test_train_command_end_to_end(self, tmp_path):
        rng = random.Random(0)
        samples, labels = [], []
        for i in range(80):
            bulk = rng.random() < 0.6
            sample = make_sample(
                str(i),
                subject="Weekly deals" if bulk else "Re: dinner",
                sender="news@e.shop.example" if bulk else "Alex <alex@gmail.com>",
                headers={"precedence": "bulk", "list-id": "<x>"} if bulk else {},
                has_list_unsubscribe=bulk,
            )
            samples.append(sample)
            labels.append(lab(str(i), "delete" if bulk else "keep", labeler="human"))
        write_samples(tmp_path / "s.jsonl", samples)
        for label in labels:
            append_jsonl(tmp_path / "l.jsonl", label)
        out = tmp_path / "combiner.json"
        result = CliRunner().invoke(
            cli,
            [
                "train",
                str(tmp_path / "s.jsonl"),
                "--labels",
                str(tmp_path / "l.jsonl"),
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Trained on 80 emails" in result.output
        combiner = Combiner.load(out)
        assert combiner.metrics["false_deletes"] == 0


class TestEvaluate:
    def test_counts_wrong_deletes_per_path(self):
        alert = make_sample(
            "1",
            subject="Direct Debit Withdrawal",
            sender="Alerts <alerts@e.bank.example>",
            body="Withdrawal posted. View in browser. Unsubscribe.",
            has_list_unsubscribe=True,
        )
        promo = make_sample(
            "2",
            subject="Big sale",
            sender="Deals <deals@e.shop.example>",
            body="Save 50%. View in browser. Unsubscribe.",
            has_list_unsubscribe=True,
        )
        code = make_sample("3", subject="Your verification code is 123456")
        personal = make_sample("4", subject="Dinner?", sender="Alex <alex@gmail.com>")
        labels = {
            "1": lab("1", "keep", loss=Loss.BAD),
            "2": lab("2", "delete"),
            "3": lab(
                "3", "keep", lifetime=Lifetime.MINUTES
            ),  # expired: effectively delete
            "4": lab("4", "keep"),
        }
        rule = RuleSet(
            rules=[Rule(match=RuleMatch.DOMAIN, value="bank.example", action="delete")]
        )
        ev = evaluate(
            [alert, promo, code, personal], [labels], Config(email="me@x"), rule
        )
        # Marketing indicators alone delete nothing: the alert and the promo
        # are left to the AI (not scored here); a bad rule still shows up
        assert "sender_pattern" not in ev.paths
        assert (ev.paths["rule"].deletes, ev.paths["rule"].wrong_deletes) == (1, 1)
        assert (
            ev.paths["expired"].deletes == 1 and ev.paths["expired"].wrong_deletes == 0
        )
        assert ev.wrong_deletes == 1 and ev.precision == 1 / 2
        assert ev.recall == 1 / 2
        assert "Direct Debit Withdrawal" in ev.wrong[0]
        assert "precision" in format_evaluation(ev)[0]
