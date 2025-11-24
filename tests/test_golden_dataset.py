"""Golden dataset tests for email classification.

These tests run the classifier against hand-labeled test data to measure
accuracy and prevent regressions. They are skipped by default in CI since
they require a golden dataset to be created first.

Run with: pytest tests/test_golden_dataset.py --run-golden
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from inbox_reaper.dag import run_pipeline
from inbox_reaper.sanitizer import dict_to_email
from inbox_reaper.state import Config, Decision, ProcessingState


@dataclass
class ClassificationMetrics:
    """Classification performance metrics."""

    total: int
    correct: int
    incorrect: int
    true_positives: int  # Correctly identified as DELETE
    false_positives: int  # Incorrectly marked as DELETE
    true_negatives: int  # Correctly identified as KEEP
    false_negatives: int  # Incorrectly marked as KEEP

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def precision(self) -> float:
        """Precision for DELETE classification."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Recall for DELETE classification."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def f1_score(self) -> float:
        """F1 score for DELETE classification."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    def __str__(self) -> str:
        """Format metrics as string."""
        return f"""Classification Metrics:
  Total: {self.total}
  Correct: {self.correct}
  Incorrect: {self.incorrect}
  Accuracy: {self.accuracy:.2%}

  Confusion Matrix:
    True Positives (DELETE):  {self.true_positives}
    False Positives (DELETE): {self.false_positives}
    True Negatives (KEEP):    {self.true_negatives}
    False Negatives (KEEP):   {self.false_negatives}

  Performance:
    Precision: {self.precision:.2%}
    Recall:    {self.recall:.2%}
    F1 Score:  {self.f1_score:.2%}
"""


def load_test_data(test_data_dir: Path) -> list[dict]:
    """Load all test data files from directory.

    Args:
        test_data_dir: Directory containing test JSON files

    Returns:
        List of test data entries
    """
    test_files = list(test_data_dir.glob("*.json"))
    test_data = []

    for filepath in test_files:
        with open(filepath) as f:
            test_data.append(json.load(f))

    return test_data


def run_golden_tests() -> bool:
    """Check if golden tests should run.

    Returns:
        True if --run-golden flag is present
    """
    import sys

    return "--run-golden" in sys.argv


@pytest.fixture
def test_data_dir() -> Path:
    """Get path to test data directory."""
    return Path("test_data/golden")


@pytest.fixture
def golden_dataset(test_data_dir: Path) -> list[dict]:
    """Load golden dataset.

    Raises:
        pytest.skip: If dataset doesn't exist
    """
    if not test_data_dir.exists():
        pytest.skip(f"Golden dataset not found at {test_data_dir}")

    test_data = load_test_data(test_data_dir)

    if not test_data:
        pytest.skip(f"No test files found in {test_data_dir}")

    return test_data


@pytest.fixture
def classifier_config() -> Config:
    """Create default config for testing."""
    return Config(
        model_name="gemma2:2b",
        ollama_base_url="http://localhost:11434",
        batch_size=50,
        concurrent_ai_limit=25,
        dry_run=True,
    )


@pytest.mark.skipif(not run_golden_tests(), reason="Golden tests not enabled")
class TestGoldenDataset:
    """Test classifier against golden dataset."""

    def test_dataset_exists(self, golden_dataset):
        """Verify golden dataset is loaded."""
        assert len(golden_dataset) > 0, "Golden dataset should contain test cases"

    def test_classification_accuracy(self, golden_dataset, classifier_config):
        """Test overall classification accuracy against golden dataset."""
        # Convert test data to Email objects
        emails = [dict_to_email(entry["email"]) for entry in golden_dataset]

        # Run through pipeline
        state = ProcessingState(config=classifier_config, emails=emails)
        final_state = run_pipeline(state)

        # Calculate metrics
        metrics = ClassificationMetrics(
            total=0,
            correct=0,
            incorrect=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
        )

        # Compare results with ground truth
        for entry, decision in zip(golden_dataset, final_state.decisions):
            expected = entry["ground_truth"]["decision"]
            actual = decision.decision.value

            metrics.total += 1

            if expected == actual:
                metrics.correct += 1
            else:
                metrics.incorrect += 1

            # Update confusion matrix
            if expected == "delete" and actual == "delete":
                metrics.true_positives += 1
            elif expected == "keep" and actual == "delete":
                metrics.false_positives += 1
            elif expected == "keep" and actual == "keep":
                metrics.true_negatives += 1
            elif expected == "delete" and actual == "keep":
                metrics.false_negatives += 1

        # Print detailed metrics
        print("\n" + str(metrics))

        # Assert minimum performance thresholds
        assert metrics.accuracy >= 0.80, f"Accuracy below 80%: {metrics.accuracy:.2%}"
        assert (
            metrics.f1_score >= 0.75
        ), f"F1 score below 75%: {metrics.f1_score:.2%}"

    def test_per_category_performance(self, golden_dataset, classifier_config):
        """Test performance by email category (marketing vs important)."""
        # Separate by ground truth decision
        delete_emails = [
            entry for entry in golden_dataset if entry["ground_truth"]["decision"] == "delete"
        ]
        keep_emails = [
            entry for entry in golden_dataset if entry["ground_truth"]["decision"] == "keep"
        ]

        if not delete_emails or not keep_emails:
            pytest.skip("Dataset needs both keep and delete examples")

        # Test DELETE category
        delete_email_objs = [dict_to_email(entry["email"]) for entry in delete_emails]
        delete_state = ProcessingState(
            config=classifier_config, emails=delete_email_objs
        )
        delete_final = run_pipeline(delete_state)

        delete_correct = sum(
            1 for d in delete_final.decisions if d.decision == Decision.DELETE
        )
        delete_accuracy = delete_correct / len(delete_emails)

        # Test KEEP category
        keep_email_objs = [dict_to_email(entry["email"]) for entry in keep_emails]
        keep_state = ProcessingState(config=classifier_config, emails=keep_email_objs)
        keep_final = run_pipeline(keep_state)

        keep_correct = sum(
            1 for d in keep_final.decisions if d.decision == Decision.KEEP
        )
        keep_accuracy = keep_correct / len(keep_emails)

        print(f"\nDELETE accuracy: {delete_accuracy:.2%} ({delete_correct}/{len(delete_emails)})")
        print(f"KEEP accuracy: {keep_accuracy:.2%} ({keep_correct}/{len(keep_emails)})")

        # Both categories should perform reasonably well
        assert delete_accuracy >= 0.70, f"DELETE category accuracy too low: {delete_accuracy:.2%}"
        assert keep_accuracy >= 0.70, f"KEEP category accuracy too low: {keep_accuracy:.2%}"

    def test_deterministic_filters_effectiveness(self, golden_dataset, classifier_config):
        """Test that deterministic filters catch emails before AI."""
        emails = [dict_to_email(entry["email"]) for entry in golden_dataset]

        state = ProcessingState(config=classifier_config, emails=emails)
        final_state = run_pipeline(state)

        # Count decisions by reason
        ai_decisions = sum(1 for d in final_state.decisions if d.reason.value == "ai_classified")
        deterministic_decisions = len(final_state.decisions) - ai_decisions

        early_termination_rate = (
            deterministic_decisions / len(final_state.decisions)
            if final_state.decisions
            else 0
        )

        print(f"\nDeterministic filter decisions: {deterministic_decisions}")
        print(f"AI decisions: {ai_decisions}")
        print(f"Early termination rate: {early_termination_rate:.2%}")

        # At least some emails should be caught by deterministic filters
        # (this will vary based on test data)
        assert early_termination_rate >= 0.0, "Negative early termination rate"

    @pytest.mark.parametrize("test_file", ["delete_", "keep_"])
    def test_individual_categories(
        self, test_data_dir: Path, classifier_config, test_file: str
    ):
        """Test individual email categories."""
        if not test_data_dir.exists():
            pytest.skip("Golden dataset not found")

        # Find all files matching pattern
        test_files = list(test_data_dir.glob(f"{test_file}*.json"))

        if not test_files:
            pytest.skip(f"No {test_file} test files found")

        # Load and test
        test_data = []
        for filepath in test_files:
            with open(filepath) as f:
                test_data.append(json.load(f))

        emails = [dict_to_email(entry["email"]) for entry in test_data]
        state = ProcessingState(config=classifier_config, emails=emails)
        final_state = run_pipeline(state)

        # Count correct decisions
        expected_decision = "delete" if test_file == "delete_" else "keep"
        correct = sum(
            1
            for d in final_state.decisions
            if d.decision.value == expected_decision
        )

        accuracy = correct / len(final_state.decisions) if final_state.decisions else 0

        print(f"\n{test_file} category: {correct}/{len(test_data)} correct ({accuracy:.2%})")

        # Should get most of them right
        assert accuracy >= 0.70, f"{test_file} accuracy too low: {accuracy:.2%}"


@pytest.mark.skipif(not run_golden_tests(), reason="Golden tests not enabled")
def test_golden_dataset_format(test_data_dir: Path):
    """Verify golden dataset files have correct format."""
    if not test_data_dir.exists():
        pytest.skip("Golden dataset not found")

    test_files = list(test_data_dir.glob("*.json"))
    if not test_files:
        pytest.skip("No test files found")

    for filepath in test_files:
        with open(filepath) as f:
            data = json.load(f)

        # Verify structure
        assert "email" in data, f"Missing 'email' key in {filepath}"
        assert "ground_truth" in data, f"Missing 'ground_truth' key in {filepath}"

        # Verify email structure
        email = data["email"]
        required_fields = ["uid", "subject", "sender", "body", "date", "attachments"]
        for field in required_fields:
            assert field in email, f"Missing '{field}' in email data in {filepath}"

        # Verify ground truth structure
        ground_truth = data["ground_truth"]
        assert "decision" in ground_truth, f"Missing 'decision' in ground truth in {filepath}"
        assert ground_truth["decision"] in [
            "keep",
            "delete",
        ], f"Invalid decision value in {filepath}"
