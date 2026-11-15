"""Tests for the trained combiner."""

import json
import random

import numpy as np
import pytest

from inbox_reaper.combiner import Combiner, TrainingExample, _thresholds, train
from inbox_reaper.state import Decision


def synthetic_examples(n: int = 200, seed: int = 0) -> list[TrainingExample]:
    """'bulk' strongly predicts delete, 'personal' predicts keep, 'noise' is random."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        delete = rng.random() < 0.7
        bulk = float(rng.random() < (0.9 if delete else 0.05))
        personal = float(rng.random() < (0.02 if delete else 0.8))
        examples.append(
            TrainingExample(
                features={"bulk": bulk, "personal": personal, "noise": rng.random()},
                label=Decision.DELETE if delete else Decision.KEEP,
            )
        )
    return examples


class TestTrain:
    def test_learns_signal_and_orders_thresholds(self):
        combiner = train(synthetic_examples())
        weights = dict(combiner.top_weights())
        assert weights["bulk"] > 0 > weights["personal"]
        assert abs(weights["noise"]) < abs(weights["bulk"])
        assert 0 <= combiner.keep_threshold < combiner.delete_threshold <= 0.999

    def test_decides_confident_cases(self):
        combiner = train(synthetic_examples())
        decision, p = combiner.decide({"bulk": 1.0, "personal": 0.0, "noise": 0.5})
        assert p > 0.5
        assert decision in {Decision.DELETE, None}
        decision, p = combiner.decide({"bulk": 0.0, "personal": 1.0, "noise": 0.5})
        assert decision == Decision.KEEP and p < combiner.keep_threshold + 1e-9

    def test_delete_threshold_clears_every_held_out_keep(self):
        examples = synthetic_examples()
        combiner = train(examples)
        # On the training data, a keep-labelled email is never confidently deleted
        # by the deployed thresholds (which came from held-out probabilities)
        keeps = [e for e in examples if e.label == Decision.KEEP]
        fitted = [combiner.probability(e.features) for e in keeps]
        assert max(fitted) < combiner.delete_threshold

    def test_metrics_are_reported(self):
        m = train(synthetic_examples()).metrics
        assert m["examples"] == 200
        assert m["keep_examples"] + m["delete_examples"] == 200
        assert 0 <= m["contested_share"] <= 1

    def test_weights_shift_the_model(self):
        examples = synthetic_examples()
        heavy = [
            e.model_copy(update={"weight": 10.0}) if e.label == Decision.KEEP else e
            for e in examples
        ]
        light, weighted = train(examples), train(heavy)
        probe = {"bulk": 1.0, "personal": 1.0, "noise": 0.5}
        assert weighted.probability(probe) < light.probability(probe)

    def test_needs_both_labels(self):
        only_delete = [TrainingExample(features={"a": 1.0}, label=Decision.DELETE)] * 5
        with pytest.raises(ValueError, match="both"):
            train(only_delete)


class TestPersistence:
    def test_round_trip_and_missing_features(self, tmp_path):
        combiner = train(synthetic_examples())
        path = tmp_path / "combiner.json"
        combiner.save(path)
        loaded = Combiner.load(path)
        probe = {
            "bulk": 1.0,
            "extra": 5.0,
        }  # missing 'personal'/'noise', unknown 'extra'
        assert loaded.probability(probe) == pytest.approx(combiner.probability(probe))

    def test_rejects_other_versions(self, tmp_path):
        path = tmp_path / "combiner.json"
        data = train(synthetic_examples()).model_dump()
        data["version"] = 99
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="version"):
            Combiner.load(path)


class TestPrecisionFirstThresholds:
    def test_zero_budget_sits_above_every_keep(self):
        p = np.array([0.2, 0.7, 0.95, 0.99, 0.999])
        y = np.array([0, 0, 1, 1, 1], dtype=float)
        lo, hi = _thresholds(p, y, np.ones(5), max_missed=0.1, max_false_delete=0.0)
        assert 0.7 < hi <= 0.95
        assert lo < hi

    def test_keep_at_the_top_disables_deletes(self):
        p = np.array([0.1, 0.99, 0.999999])
        y = np.array([1, 1, 0], dtype=float)
        _, hi = _thresholds(p, y, np.ones(3), max_missed=0.1, max_false_delete=0.0)
        assert hi == 1.0  # nothing can score >= 1.0: the combiner never deletes

    def test_budget_counts_weight_not_emails(self):
        p = np.array([0.5, 0.9, 0.95, 0.99])
        y = np.array([0, 0, 1, 1], dtype=float)
        # A 10% budget tolerates the 0.9 keep at weight 1, but not at weight 10
        light = _thresholds(p, y, np.array([1, 1, 1, 1.0]), 0.1, max_false_delete=0.5)[
            1
        ]
        heavy = _thresholds(p, y, np.array([1, 10, 1, 1.0]), 0.1, max_false_delete=0.5)[
            1
        ]
        assert light <= 0.9 < heavy

    def test_metrics_report_precision_and_recall(self):
        m = train(synthetic_examples()).metrics
        assert 0 <= m["delete_recall"] <= 1
        assert m["delete_precision"] == pytest.approx(
            m["deletes_caught"] / (m["deletes_caught"] + m["false_deletes"])
            if m["deletes_caught"] + m["false_deletes"]
            else 1.0
        )
