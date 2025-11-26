"""Unit tests for MLX JSON parsing in classify_with_ai.

Tests the various JSON response formats that MLX models might return,
including markdown code fences, extra text, and error cases.
"""

import unittest
from datetime import datetime
from unittest.mock import patch

from inbox_reaper.agents import classify_with_ai
from inbox_reaper.state import Config, Decision, Email, FilterReason


class TestMLXJSONParsing(unittest.TestCase):
    """Test suite for MLX JSON parsing in classify_with_ai function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_email = Email(
            uid="test-123",
            subject="Test Email",
            sender="test@example.com",
            body="This is a test email body.",
            date=datetime.now(),
            attachments=[],
        )

        self.test_config = Config(
            email="user@example.com",
            model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
            batch_size=10,
            concurrent_ai_limit=5,
            dry_run=True,
            keywords=[],
            whitelist_domains=[],
            ai_confidence_threshold=0.7,
        )

    @patch("inbox_reaper.agents.generate_text")
    def test_plain_json_response(self, mock_generate_text):
        """Test parsing plain JSON response without any wrapping."""
        # MLX returns clean JSON
        mock_generate_text.return_value = '{"is_marketing": true, "confidence": 0.95}'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should be DELETE because is_marketing=true and confidence >= 0.7
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.reason, FilterReason.AI_CLASSIFIED)

    @patch("inbox_reaper.agents.generate_text")
    def test_json_with_markdown_code_fence(self, mock_generate_text):
        """Test parsing JSON wrapped in markdown code fence with language."""
        # MLX returns JSON in markdown code fence
        mock_generate_text.return_value = """```json
{"is_marketing": true, "confidence": 0.85}
```"""

        result = classify_with_ai(self.test_email, self.test_config)

        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.85)

    @patch("inbox_reaper.agents.generate_text")
    def test_json_with_plain_code_fence(self, mock_generate_text):
        """Test parsing JSON wrapped in plain markdown code fence."""
        # MLX returns JSON in plain code fence without language specifier
        mock_generate_text.return_value = """```
{"is_marketing": false, "confidence": 0.60}
```"""

        result = classify_with_ai(self.test_email, self.test_config)

        # Should be KEEP because is_marketing=false
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.60)

    @patch("inbox_reaper.agents.generate_text")
    def test_json_with_surrounding_text(self, mock_generate_text):
        """Test parsing JSON with extra text before and after."""
        # MLX adds explanatory text around JSON
        mock_generate_text.return_value = """Here is my analysis:
{"is_marketing": true, "confidence": 0.92}
This email appears to be promotional."""

        result = classify_with_ai(self.test_email, self.test_config)

        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.92)

    @patch("inbox_reaper.agents.generate_text")
    def test_json_with_whitespace(self, mock_generate_text):
        """Test parsing JSON with extra whitespace."""
        # MLX returns JSON with lots of whitespace
        mock_generate_text.return_value = """

        {"is_marketing": false, "confidence": 0.45}

        """

        result = classify_with_ai(self.test_email, self.test_config)

        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.45)

    @patch("inbox_reaper.agents.generate_text")
    def test_low_confidence_marketing(self, mock_generate_text):
        """Test that low confidence marketing emails are kept (safe default)."""
        # MLX says marketing but with low confidence
        mock_generate_text.return_value = '{"is_marketing": true, "confidence": 0.5}'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should be KEEP because confidence (0.5) < threshold (0.7)
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.5)

    @patch("inbox_reaper.agents.generate_text")
    def test_high_confidence_not_marketing(self, mock_generate_text):
        """Test that high confidence non-marketing emails are kept."""
        # MLX says not marketing with high confidence
        mock_generate_text.return_value = '{"is_marketing": false, "confidence": 0.99}'

        result = classify_with_ai(self.test_email, self.test_config)

        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.99)

    @patch("inbox_reaper.agents.generate_text")
    def test_exact_threshold_confidence(self, mock_generate_text):
        """Test behavior at exact confidence threshold."""
        # MLX returns confidence exactly at threshold
        mock_generate_text.return_value = '{"is_marketing": true, "confidence": 0.7}'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should DELETE because confidence >= threshold (0.7 >= 0.7)
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.7)

    @patch("inbox_reaper.agents.generate_text")
    def test_invalid_json_syntax(self, mock_generate_text):
        """Test handling of invalid JSON syntax."""
        # MLX returns malformed JSON
        mock_generate_text.return_value = '{"is_marketing": true, "confidence": 0.9'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should default to KEEP with 0 confidence on error
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.reason, FilterReason.AI_CLASSIFIED)

    @patch("inbox_reaper.agents.generate_text")
    def test_no_json_in_response(self, mock_generate_text):
        """Test handling of response with no JSON."""
        # MLX returns text without JSON
        mock_generate_text.return_value = "I cannot classify this email."

        result = classify_with_ai(self.test_email, self.test_config)

        # Should default to KEEP with 0 confidence
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.0)

    @patch("inbox_reaper.agents.generate_text")
    def test_json_missing_required_fields(self, mock_generate_text):
        """Test handling of JSON missing required fields."""
        # MLX returns JSON but missing required fields
        mock_generate_text.return_value = '{"is_marketing": true}'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should default to KEEP with 0 confidence on validation error
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.0)

    @patch("inbox_reaper.agents.generate_text")
    def test_json_extra_fields(self, mock_generate_text):
        """Test handling of JSON with extra fields (should be ignored)."""
        # MLX returns JSON with extra fields
        mock_generate_text.return_value = (
            """{"is_marketing": true, "confidence": 0.88, """
            """"reasoning": "Contains promotional language"}"""
        )

        result = classify_with_ai(self.test_email, self.test_config)

        # Should work fine, extra fields are ignored
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.88)

    @patch("inbox_reaper.agents.generate_text")
    def test_json_wrong_types(self, mock_generate_text):
        """Test handling of JSON with wrong field types."""
        # MLX returns JSON with wrong types
        mock_generate_text.return_value = (
            '{"is_marketing": "yes", "confidence": "high"}'
        )

        result = classify_with_ai(self.test_email, self.test_config)

        # Should default to KEEP with 0 confidence on validation error
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.0)

    @patch("inbox_reaper.agents.generate_text")
    def test_multiple_json_objects(self, mock_generate_text):
        """Test handling of response with multiple JSON objects on separate lines."""
        # MLX returns multiple JSON objects (json.loads will fail)
        mock_generate_text.return_value = (
            """First analysis: {"is_marketing": false, "confidence": 0.3}\n"""
            """        Second analysis: {"is_marketing": true, "confidence": 0.9}"""
        )

        result = classify_with_ai(self.test_email, self.test_config)

        # Should default to KEEP because json.loads fails on multiple objects
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.0)

    @patch("inbox_reaper.agents.generate_text")
    def test_nested_json_objects(self, mock_generate_text):
        """Test handling of nested JSON objects."""
        # MLX returns nested JSON (outer object should be used)
        mock_generate_text.return_value = (
            """{"is_marketing": true, "confidence": 0.82, """
            """"details": {"reason": "promotional"}}"""
        )

        result = classify_with_ai(self.test_email, self.test_config)

        # Should parse outer object successfully
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.82)

    @patch("inbox_reaper.agents.generate_text")
    def test_markdown_fence_incomplete(self, mock_generate_text):
        """Test handling of incomplete markdown code fence."""
        # MLX returns opening fence but no closing fence
        mock_generate_text.return_value = """```json
{"is_marketing": true, "confidence": 0.75}"""

        result = classify_with_ai(self.test_email, self.test_config)

        # Should still extract the JSON successfully
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.75)

    @patch("inbox_reaper.agents.generate_text")
    def test_confidence_boundary_below_threshold(self, mock_generate_text):
        """Test confidence just below threshold."""
        # MLX returns confidence 0.69 (just below 0.7 threshold)
        mock_generate_text.return_value = '{"is_marketing": true, "confidence": 0.69}'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should KEEP because 0.69 < 0.7
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.69)

    @patch("inbox_reaper.agents.generate_text")
    def test_confidence_boundary_above_threshold(self, mock_generate_text):
        """Test confidence just above threshold."""
        # MLX returns confidence 0.71 (just above 0.7 threshold)
        mock_generate_text.return_value = '{"is_marketing": true, "confidence": 0.71}'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should DELETE because 0.71 >= 0.7
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.71)

    @patch("inbox_reaper.agents.generate_text")
    def test_zero_confidence(self, mock_generate_text):
        """Test handling of zero confidence."""
        # MLX returns 0 confidence
        mock_generate_text.return_value = '{"is_marketing": true, "confidence": 0.0}'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should KEEP because 0.0 < 0.7
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.0)

    @patch("inbox_reaper.agents.generate_text")
    def test_max_confidence(self, mock_generate_text):
        """Test handling of maximum confidence."""
        # MLX returns max confidence
        mock_generate_text.return_value = '{"is_marketing": true, "confidence": 1.0}'

        result = classify_with_ai(self.test_email, self.test_config)

        # Should DELETE because 1.0 >= 0.7
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 1.0)

    @patch("inbox_reaper.agents.generate_text")
    def test_json_array_response(self, mock_generate_text):
        """Test handling of JSON array - parser extracts object from array."""
        # MLX returns JSON array instead of object
        mock_generate_text.return_value = '[{"is_marketing": true, "confidence": 0.8}]'

        result = classify_with_ai(self.test_email, self.test_config)

        # Parser finds first { and extracts the object successfully
        # This is actually robust behavior that handles arrays gracefully
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.8)

    @patch("inbox_reaper.agents.generate_text")
    def test_unicode_in_json(self, mock_generate_text):
        """Test handling of unicode characters in JSON."""
        # MLX returns JSON with unicode
        mock_generate_text.return_value = (
            '{"is_marketing": true, "confidence": 0.85, "note": "Contains 日本語"}'
        )

        result = classify_with_ai(self.test_email, self.test_config)

        # Should parse successfully
        self.assertEqual(result.decision, Decision.DELETE)
        self.assertEqual(result.confidence, 0.85)

    @patch("inbox_reaper.agents.generate_text")
    def test_escaped_characters_in_json(self, mock_generate_text):
        """Test handling of escaped characters in JSON."""
        # MLX returns JSON with escaped characters
        mock_generate_text.return_value = (
            r'{"is_marketing": false, "confidence": 0.5, "note": "Line1\nLine2"}'
        )

        result = classify_with_ai(self.test_email, self.test_config)

        # Should parse successfully
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
