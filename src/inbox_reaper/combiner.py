"""Trained combiner: logistic regression over the feature vector.

The combiner turns a :data:`~inbox_reaper.features.FeatureVector` into
P(delete) and decides only when it is confident:

- ``p >= delete_threshold`` -> DELETE
- ``p <= keep_threshold``   -> KEEP
- otherwise                 -> contested (``None``), left to the AI classifier

Thresholds come from cross-validation on the training labels: the delete
threshold sits above every held-out probability of a mail labelled *keep*, so
on the training data the combiner never deletes something a human kept.
Examples can carry weights, so mail whose loss would hurt counts more.
"""

import json
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, Field

from .features import FeatureVector
from .state import Decision
from .timeutil import utcnow

COMBINER_VERSION = 1


class TrainingExample(BaseModel):
    features: FeatureVector
    label: Decision
    weight: float = 1.0


class Combiner(BaseModel):
    version: int = COMBINER_VERSION
    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    weights: list[float]
    bias: float
    delete_threshold: float
    keep_threshold: float
    trained_on: int
    generated_at: str = Field(default_factory=lambda: utcnow().isoformat())
    metrics: dict[str, float] = Field(default_factory=dict)

    def vector(self, features: FeatureVector) -> np.ndarray:
        """Features in training order; missing ones count as 0, extras are ignored."""
        return np.array([features.get(name, 0.0) for name in self.feature_names])

    def probability(self, features: FeatureVector) -> float:
        z = (self.vector(features) - np.array(self.mean)) / np.array(self.scale)
        return float(_sigmoid(z @ np.array(self.weights) + self.bias))

    def decide(self, features: FeatureVector) -> tuple[Decision | None, float]:
        """(decision, P(delete)); decision is None when contested."""
        p = self.probability(features)
        if p >= self.delete_threshold:
            return Decision.DELETE, p
        if p <= self.keep_threshold:
            return Decision.KEEP, p
        return None, p

    def top_weights(self, n: int = 10) -> list[tuple[str, float]]:
        pairs = zip(self.feature_names, self.weights, strict=True)
        return sorted(pairs, key=lambda p: -abs(p[1]))[:n]

    @classmethod
    def load(cls, path: str | Path) -> Self:
        model = cls.model_validate_json(Path(path).read_text())
        if model.version != COMBINER_VERSION:
            raise ValueError(
                f"{path}: combiner version {model.version}, expected {COMBINER_VERSION}"
            )
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(), indent=2) + "\n")


def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _fit(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, steps: int = 2000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Weighted L2 logistic regression by gradient descent on standardized x."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0  # constant features contribute nothing
    z = (x - mean) / scale
    coef = np.zeros(z.shape[1])
    bias = 0.0
    total = w.sum()
    for _ in range(steps):
        err = (_sigmoid(z @ coef + bias) - y) * w
        coef -= 0.5 * (z.T @ err / total + l2 * coef / total)
        bias -= 0.5 * err.sum() / total
    return mean, scale, coef, float(bias)


def _predict(
    params: tuple[np.ndarray, np.ndarray, np.ndarray, float], x: np.ndarray
) -> np.ndarray:
    mean, scale, coef, bias = params
    return _sigmoid(((x - mean) / scale) @ coef + bias)


def _thresholds(
    p: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    max_missed: float,
    max_false_delete: float,
) -> tuple[float, float]:
    """(keep_threshold, delete_threshold) for probabilities ``p`` and labels ``y``.

    The delete threshold sits just above the highest-scoring keep-labelled
    emails until the weight of keeps left above it fits ``max_false_delete``
    (a share of total keep weight; 0 means above every keep). It can reach
    1.0, which disables confident deletes entirely.
    """
    keep_p, keep_w = p[y == 0], w[y == 0]
    budget = max_false_delete * keep_w.sum()
    delete_threshold = 0.0
    above = 0.0
    for i in np.argsort(-keep_p):
        above += keep_w[i]
        if above > budget:
            delete_threshold = float(min(1.0, keep_p[i] + 1e-6))
            break
    keep_threshold = 0.0
    for t in np.unique(p):
        bucket = p <= t
        if (w * y)[bucket].sum() / w[bucket].sum() <= max_missed:
            keep_threshold = float(t)
    return min(keep_threshold, delete_threshold - 1e-6), delete_threshold


def _cv_probabilities(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, folds: int, seed: int
) -> np.ndarray:
    """Held-out P(delete) for every row, by k-fold cross-validation."""
    held_out = np.zeros(len(y))
    order = np.random.default_rng(seed).permutation(len(y))
    k = max(2, min(folds, len(y)))
    for i in range(k):
        test = order[i::k]
        train_idx = np.setdiff1d(order, test)
        if len(set(y[train_idx])) < 2:  # degenerate fold: the prior
            held_out[test] = y[train_idx].mean()
            continue
        held_out[test] = _predict(
            _fit(x[train_idx], y[train_idx], w[train_idx], l2), x[test]
        )
    return held_out


def train(
    examples: list[TrainingExample],
    *,
    l2: float = 1.0,
    folds: int = 5,
    max_missed: float = 0.1,
    max_false_delete: float = 0.0,
    seed: int = 0,
) -> Combiner:
    """Fit a combiner and pick precision-first thresholds by cross-validation.

    Args:
        examples: Labelled feature vectors (both labels must be present)
        l2: L2 regularization strength
        folds: Cross-validation folds used for thresholds and metrics
        max_missed: Largest weighted share of delete-labelled mail allowed in
            the confident-keep bucket (a missed delete only leaves mail behind)
        max_false_delete: Share of keep weight the delete threshold may still
            confidently delete. The default 0 puts the threshold above every
            held-out keep: precision first, recall is what gives way.
        seed: Fold shuffling seed

    The reported metrics are nested cross-validation: each outer fold picks its
    thresholds from held-out probabilities on its own training part only.
    """
    labels = {e.label for e in examples}
    if labels != {Decision.KEEP, Decision.DELETE}:
        raise ValueError("training needs both keep and delete examples")

    names = sorted({name for e in examples for name in e.features})
    x = np.array([[e.features.get(n, 0.0) for n in names] for e in examples])
    y = np.array([e.label == Decision.DELETE for e in examples], dtype=float)
    w = np.array([e.weight for e in examples], dtype=float)

    held_out = np.zeros(len(examples))
    decided_delete = np.zeros(len(examples), dtype=bool)
    decided_keep = np.zeros(len(examples), dtype=bool)
    order = np.random.default_rng(seed).permutation(len(examples))
    k = max(2, min(folds, len(examples)))
    for i in range(k):
        test = order[i::k]
        tr = np.setdiff1d(order, test)
        if len(set(y[tr])) < 2:  # degenerate fold: undecided
            held_out[test] = y[tr].mean()
            continue
        held_out[test] = _predict(_fit(x[tr], y[tr], w[tr], l2), x[test])
        inner = _cv_probabilities(x[tr], y[tr], w[tr], l2, folds, seed + i + 1)
        lo, hi = _thresholds(inner, y[tr], w[tr], max_missed, max_false_delete)
        decided_delete[test] = held_out[test] >= hi
        decided_keep[test] = held_out[test] <= lo

    # Deployed thresholds: from the pooled held-out probabilities of all data
    keep_threshold, delete_threshold = _thresholds(
        held_out, y, w, max_missed, max_false_delete
    )

    deletes = y == 1
    true_deletes = float((decided_delete & deletes).sum())
    false_deletes = float((decided_delete & ~deletes).sum())
    metrics = {
        "examples": float(len(examples)),
        "keep_examples": float((~deletes).sum()),
        "delete_examples": float(deletes.sum()),
        "false_deletes": false_deletes,
        "deletes_caught": true_deletes,
        "delete_precision": (
            true_deletes / (true_deletes + false_deletes)
            if true_deletes + false_deletes
            else 1.0
        ),
        "delete_recall": true_deletes / max(1.0, float(deletes.sum())),
        "missed_deletes_kept": float((decided_keep & deletes).sum()),
        "contested_share": float((~decided_delete & ~decided_keep).mean()),
    }

    mean, scale, coef, bias = _fit(x, y, w, l2)
    return Combiner(
        feature_names=names,
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=coef.tolist(),
        bias=bias,
        delete_threshold=delete_threshold,
        keep_threshold=keep_threshold,
        trained_on=len(examples),
        metrics=metrics,
    )
