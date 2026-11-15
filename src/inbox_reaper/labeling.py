"""Sampling, multi-dimensional labelling, label comparison and combiner training.

Files are JSON Lines and stay on your machine:

- samples: one ``{"uid", "sampled_at", "email": {...}}`` per line (from ``sample``)
- labels: one ``{"uid", "labeler", "action", "lifetime", "category", "loss",
  "labeled_at", "note"}`` per line (from ``label``, or any other labeller)

Each labeller writes its own labels file, so human and model labels can be
compared (``compare-labels``) and merged for training with the human first.
"""

import json
import random
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import click
from pydantic import BaseModel, Field

from .combiner import Combiner, TrainingExample, train
from .features import TTL, FeatureContext, Lifetime, email_age, extract_features
from .rules import RuleSet
from .state import Config, Decision, Email
from .text_utils import compact_urls
from .timeutil import to_utc, utcnow


class Loss(StrEnum):
    """How bad it would be if this email were deleted."""

    NONE = "none"
    ANNOYING = "annoying"
    BAD = "bad"


# Training weight of a keep-labelled email by how bad losing it would be
LOSS_WEIGHT = {Loss.NONE: 1.0, Loss.ANNOYING: 3.0, Loss.BAD: 10.0}

CATEGORIES = (
    "personal", "work", "receipt", "financial", "shipping", "travel",
    "security", "medical", "government", "promo", "newsletter",
    "notification", "github", "survey", "other",
)  # fmt: skip


class Sample(BaseModel):
    uid: str
    sampled_at: datetime
    email: Email


class Label(BaseModel):
    uid: str
    labeler: str
    action: Decision
    lifetime: Lifetime = Lifetime.FOREVER
    category: str = "other"
    loss: Loss = Loss.NONE
    labeled_at: datetime = Field(default_factory=utcnow)
    note: str = ""


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_jsonl(path: str | Path, record: BaseModel) -> None:
    with Path(path).open("a") as f:
        f.write(record.model_dump_json() + "\n")


def load_samples(path: str | Path) -> list[Sample]:
    return [Sample.model_validate(r) for r in read_jsonl(path)]


def load_labels(path: str | Path) -> dict[str, Label]:
    """Labels by uid; a later line for the same uid replaces an earlier one."""
    return {r["uid"]: Label.model_validate(r) for r in read_jsonl(path)}


# ---------------------------------------------------------------------------
# Labelling order
# ---------------------------------------------------------------------------


def review_order(
    samples: list[Sample], references: list[dict[str, Label]], seed: int = 0
) -> list[Sample]:
    """Emails where reference labellers disagree come first, then the rest.

    Disagreement on keep/delete ranks above disagreement on lifetime only.
    Within each tier the order is random (seeded) to avoid bias.
    """
    rng = random.Random(seed)

    def tier(sample: Sample) -> int:
        labels = [ref[sample.uid] for ref in references if sample.uid in ref]
        if len({lab.action for lab in labels}) > 1:
            return 0
        if len({lab.lifetime for lab in labels}) > 1:
            return 1
        return 2

    shuffled = samples[:]
    rng.shuffle(shuffled)
    return sorted(shuffled, key=tier)


# ---------------------------------------------------------------------------
# Interactive labelling
# ---------------------------------------------------------------------------


def _age(sample: Sample) -> str:
    seconds = (
        sample.sampled_at - to_utc(sample.email.date, naive_is_local=False)
    ).total_seconds()
    for unit, size in (("days", 86400), ("hours", 3600), ("minutes", 60)):
        if seconds >= size:
            return f"{seconds / size:.0f} {unit}"
    return "just now"


def show_email(sample: Sample, index: int, total: int, preview: int = 900) -> None:
    e = sample.email
    click.clear()
    click.secho(f"[{index}/{total}]  uid {e.uid}", dim=True)
    click.secho(f"From:    {e.sender}", bold=True)
    click.secho(f"Subject: {e.subject}", bold=True)
    click.echo(f"Date:    {e.date:%Y-%m-%d %H:%M}  ({_age(sample)} before sampling)")
    if e.attachments:
        click.echo(f"Attachments: {', '.join(e.attachments)}")
    click.echo("-" * 72)
    body = compact_urls(" ".join(e.body.split()))
    click.echo(body[:preview] + (" ..." if len(body) > preview else ""))
    click.echo("-" * 72)


def _choice(prompt: str, options: dict[str, str]) -> str:
    # Every option starts with its key: "[k]eep/[d]elete"
    keys = "/".join(f"[{k}]{v[1:]}" for k, v in options.items())
    answer = click.prompt(
        f"{prompt} {keys}", type=click.Choice(list(options)), show_choices=False
    )
    return options[answer]


def ask_label(sample: Sample, labeler: str) -> Label | None:
    """Ask every dimension for one email. Returns None to skip; raises Abort to quit."""
    action = _choice(
        "Keep or delete?", {"k": "keep", "d": "delete", "s": "skip", "q": "quit"}
    )
    if action == "quit":
        raise click.Abort
    if action == "skip":
        return None
    lifetime = _choice("Useful for?", {"m": "minutes", "d": "days", "f": "forever"})
    numbered = {str(i): c for i, c in enumerate(CATEGORIES, 1)}
    click.echo("  " + "  ".join(f"{i}={c}" for i, c in numbered.items()))
    category = numbered[
        click.prompt("Category", type=click.Choice(list(numbered)), show_choices=False)
    ]
    loss = _choice(
        "If this were deleted it would be", {"n": "none", "a": "annoying", "b": "bad"}
    )
    note = click.prompt("Note (Enter to skip)", default="", show_default=False)
    return Label(
        uid=sample.uid,
        labeler=labeler,
        action=Decision(action),
        lifetime=Lifetime(lifetime),
        category=category,
        loss=Loss(loss),
        note=note,
    )


def label_interactively(
    samples: list[Sample], labels_path: str | Path, labeler: str, limit: int | None
) -> int:
    """Label unlabelled samples in order, saving after each one. Returns the count."""
    done = load_labels(labels_path)
    todo = [s for s in samples if s.uid not in done][:limit]
    count = 0
    try:
        for i, sample in enumerate(todo, 1):
            show_email(sample, i, len(todo))
            if label := ask_label(sample, labeler):
                append_jsonl(labels_path, label)
                count += 1
    except click.Abort:
        click.echo("\nStopped; progress is saved.")
    return count


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(label_sets: dict[str, dict[str, Label]]) -> list[str]:
    """Pairwise agreement per dimension, as printable lines."""
    lines = []
    names = list(label_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(label_sets[a]) & set(label_sets[b]))
            if not shared:
                lines.append(f"{a} vs {b}: no emails in common")
                continue
            la, lb = label_sets[a], label_sets[b]
            lines.append(f"{a} vs {b} ({len(shared)} emails in common)")
            for dim in ("action", "lifetime", "category"):
                agree = sum(getattr(la[u], dim) == getattr(lb[u], dim) for u in shared)
                lines.append(f"  {dim:9} agreement {agree / len(shared):5.0%}")
            confusion = Counter(
                (la[u].action.value, lb[u].action.value) for u in shared
            )
            lines.append(
                f"  {a} keep / {b} delete: {confusion['keep', 'delete']}   "
                f"{a} delete / {b} keep: {confusion['delete', 'keep']}"
            )
    return lines


def disagreements(
    label_sets: dict[str, dict[str, Label]], samples: dict[str, Sample]
) -> Iterable[str]:
    """One line per email whose labellers disagree on keep/delete."""
    uids = set().union(*label_sets.values())
    for uid in sorted(uids, key=lambda u: int(u) if u.isdigit() else 0):
        votes = {n: s[uid] for n, s in label_sets.items() if uid in s}
        if len({lab.action for lab in votes.values()}) > 1:
            subject = samples[uid].email.subject[:50] if uid in samples else ""
            detail = ", ".join(
                f"{n}={lab.action.value}/{lab.lifetime.value}"
                for n, lab in votes.items()
            )
            yield f"{uid:>8}  {subject:50}  {detail}"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def effective_action(label: Label, sample: Sample) -> Decision:
    """The label's action with its lifetime applied as of sampling time.

    A keep whose useful life had already run out when the email was sampled
    (a code over an hour old, a delivery notice over 30 days old) counts as a
    delete, so the time rule is learned even from labellers who judged the
    email's type rather than its age.
    """
    ttl = TTL.get(label.lifetime)
    if (
        label.action == Decision.KEEP
        and ttl
        and email_age(sample.email, sample.sampled_at) > ttl
    ):
        return Decision.DELETE
    return label.action


def training_examples(
    samples: list[Sample],
    label_sets: list[dict[str, Label]],
    config: Config,
    rules: RuleSet | None = None,
) -> list[TrainingExample]:
    """Join samples with labels (earlier label sets take precedence).

    Features are computed as of ``sampled_at`` so time-sensitivity matches
    what the labeller saw.
    """
    examples = []
    for sample in samples:
        label = next((s[sample.uid] for s in label_sets if sample.uid in s), None)
        if label is None:
            continue
        ctx = FeatureContext(config=config, rules=rules, now=sample.sampled_at)
        action = effective_action(label, sample)
        weight = LOSS_WEIGHT[label.loss] if action == Decision.KEEP else 1.0
        examples.append(
            TrainingExample(
                features=extract_features(sample.email, ctx),
                label=action,
                weight=weight,
            )
        )
    return examples


def train_combiner(
    samples: list[Sample],
    label_sets: list[dict[str, Label]],
    config: Config,
    rules: RuleSet | None = None,
) -> Combiner:
    return train(training_examples(samples, label_sets, config, rules))


# ---------------------------------------------------------------------------
# Evaluation: precision and recall per deletion path
# ---------------------------------------------------------------------------


class PathStats(BaseModel):
    """Outcomes for one decision path (e.g. expiry, heuristics, combiner)."""

    deletes: int = 0
    wrong_deletes: int = 0  # deleted mail labelled keep: the unforgivable error
    keeps: int = 0
    missed_deletes: int = 0  # kept mail labelled delete: harmless

    @property
    def precision(self) -> float:
        return (
            (self.deletes - self.wrong_deletes) / self.deletes if self.deletes else 1.0
        )


class Evaluation(BaseModel):
    name: str
    paths: dict[str, PathStats]
    to_ai: int
    total: int
    delete_labels: int
    wrong: list[str] = Field(default_factory=list)  # "path: subject" of wrong deletes

    @property
    def deletes(self) -> int:
        return sum(p.deletes for p in self.paths.values())

    @property
    def wrong_deletes(self) -> int:
        return sum(p.wrong_deletes for p in self.paths.values())

    @property
    def precision(self) -> float:
        return (
            (self.deletes - self.wrong_deletes) / self.deletes if self.deletes else 1.0
        )

    @property
    def recall(self) -> float:
        return (self.deletes - self.wrong_deletes) / max(1, self.delete_labels)


def evaluate(
    samples: list[Sample],
    label_sets: list[dict[str, Label]],
    config: Config,
    rules: RuleSet | None = None,
    combiner: Combiner | None = None,
) -> Evaluation:
    """Replay the deterministic stages (and the combiner, if given) on labelled mail.

    Emails left to the AI are counted as ``to_ai``; the AI is not run. Each
    email is judged as of its ``sampled_at`` time, against its effective label.
    """
    # Deferred: imports LangGraph
    from .langgraph_streaming import classify_deterministic  # noqa: PLC0415

    paths: dict[str, PathStats] = {}
    wrong: list[str] = []
    to_ai = total = delete_labels = 0
    for sample in samples:
        label = next((s[sample.uid] for s in label_sets if sample.uid in s), None)
        if label is None:
            continue
        total += 1
        truth = effective_action(label, sample)
        delete_labels += truth == Decision.DELETE
        email = sample.email
        decision = classify_deterministic(
            email, config, rules, heuristics=combiner is None, now=sample.sampled_at
        )
        verdict: Decision | None = None
        path = "ai"
        if decision is not None:
            verdict, path = decision.decision, decision.reason.value
        elif combiner is not None:
            ctx = FeatureContext(config=config, rules=rules, now=sample.sampled_at)
            verdict, _ = combiner.decide(extract_features(email, ctx))
            path = "combiner"
        if verdict is None:
            to_ai += 1
            continue
        stats = paths.setdefault(path, PathStats())
        if verdict == Decision.DELETE:
            stats.deletes += 1
            if truth == Decision.KEEP:
                stats.wrong_deletes += 1
                wrong.append(f"{path}: {email.subject[:60]} ({email.sender[:40]})")
        else:
            stats.keeps += 1
            stats.missed_deletes += truth == Decision.DELETE
    return Evaluation(
        name="with combiner" if combiner else "today (heuristics)",
        paths=paths,
        to_ai=to_ai,
        total=total,
        delete_labels=delete_labels,
        wrong=wrong,
    )


def format_evaluation(ev: Evaluation) -> list[str]:
    lines = [
        f"{ev.name}: delete precision {ev.precision:.1%} "
        f"({ev.wrong_deletes} wrong of {ev.deletes} deletes), recall {ev.recall:.1%}, "
        f"{ev.to_ai}/{ev.total} left to the AI"
    ]
    for name, p in sorted(ev.paths.items(), key=lambda kv: -kv[1].deletes):
        lines.append(
            f"  {name:16} deletes {p.deletes:>4} (wrong {p.wrong_deletes:>3}, "
            f"precision {p.precision:6.1%})   keeps {p.keeps:>4} "
            f"(of which deletable {p.missed_deletes})"
        )
    return lines
