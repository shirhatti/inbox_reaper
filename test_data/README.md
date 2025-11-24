# Golden Test Dataset

This directory contains hand-labeled test emails for benchmarking the classifier.

## Creating Test Data

Use the `create-test-data` command to fetch real emails and create sanitized test cases:

```bash
# Fetch 20 emails and interactively label them
inbox-reaper create-test-data user@gmail.com --count 20

# Specify custom output directory
inbox-reaper create-test-data user@gmail.com --count 50 --output-dir test_data/golden
```

### Interactive Process

For each email, the tool will:

1. **Show email preview** - Display sender, subject, date, and body snippet
2. **Ask for inclusion** - Choose whether to include this email in the dataset
3. **Auto-sanitize PII** - Automatically redact emails, phone numbers, credit cards, SSNs, etc.
4. **Review diff in $EDITOR** - See what was changed (original vs sanitized)
5. **Optional manual edits** - Make additional changes to the sanitized version
6. **Label the email** - Classify as "keep" or "delete" with reasoning
7. **Save** - Store as JSON file in `test_data/golden/`

### PII Sanitization

The sanitizer automatically handles:

- **Email addresses**: Deterministic hashing (preserves patterns)
- **Domains**: Common domains preserved (gmail.com), others anonymized
- **Phone numbers**: Redacted as `[PHONE_REDACTED]`
- **Credit cards**: Luhn-validated, redacted as `[CC_REDACTED]`
- **SSNs**: Redacted as `[SSN_REDACTED]`
- **IP addresses**: Redacted as `[IP_REDACTED]`
- **URLs**: Paths redacted, domains preserved for pattern detection

## Test Data Format

Each test file contains:

```json
{
  "email": {
    "uid": "test_001",
    "subject": "SALE: 50% off at RETAILER_A",
    "sender": "marketing@retailer_a.example",
    "body": "Limited time offer! ...",
    "date": "2024-01-15T10:30:00",
    "attachments": ["promo.jpg"]
  },
  "ground_truth": {
    "decision": "delete",
    "reason": "marketing newsletter",
    "notes": "Promotional email with unsubscribe link",
    "created_at": "2024-11-24T10:30:00",
    "created_from_account": "user@gmail.com"
  }
}
```

### Filename Convention

Files are named: `{decision}_{uid}_{index}.json`

Examples:
- `delete_1234_001.json` - Email that should be deleted
- `keep_5678_002.json` - Email that should be kept

## Running Golden Dataset Tests

Golden dataset tests are **skipped by default** in CI. To run them:

```bash
# Run golden dataset tests
pytest tests/test_golden_dataset.py --run-golden

# Run with verbose output
pytest tests/test_golden_dataset.py --run-golden -v

# Run specific test
pytest tests/test_golden_dataset.py::TestGoldenDataset::test_classification_accuracy --run-golden
```

### Test Coverage

The test suite includes:

1. **Overall accuracy** - Measures precision, recall, F1 score
2. **Per-category performance** - Tests DELETE vs KEEP separately
3. **Deterministic filter effectiveness** - How many emails bypass AI
4. **Individual categories** - Validates each decision type
5. **Format validation** - Ensures test files are properly structured

### Performance Thresholds

Default assertions:
- **Accuracy**: ≥ 80%
- **F1 Score**: ≥ 75%
- **Category Accuracy**: ≥ 70% for both keep/delete

## Directory Structure

```
test_data/
├── README.md                 # This file
└── golden/                   # Hand-labeled test cases
    ├── delete_*.json         # Emails that should be deleted
    ├── keep_*.json           # Emails that should be kept
    └── .gitignore            # Exclude test files from git
```

## Best Practices

### Dataset Size

- **Minimum**: 50 emails (25 keep, 25 delete)
- **Recommended**: 200+ emails for robust testing
- **Ideal**: 500+ emails with diverse patterns

### Label Distribution

Aim for balanced representation:
- Marketing/promotional emails
- Bills and financial statements
- Personal correspondence
- Newsletters (both wanted and unwanted)
- Transactional emails
- Edge cases (ambiguous classifications)

### Diversity

Include variety in:
- Email length (short, medium, long)
- HTML vs plain text
- With/without attachments
- Different sender patterns
- Various promotional language styles
- International characters and languages

### Review Process

When labeling:
1. **Be consistent** - Use the same criteria across all emails
2. **Document edge cases** - Add notes for ambiguous decisions
3. **Verify sanitization** - Check that no PII leaked through
4. **Test incrementally** - Add 10-20 emails, run tests, repeat

## Regression Testing

Use golden dataset tests to prevent performance degradation:

```bash
# Run before making changes (establish baseline)
pytest tests/test_golden_dataset.py --run-golden > baseline.txt

# Make changes to classifier
# ...

# Run again and compare
pytest tests/test_golden_dataset.py --run-golden > current.txt
diff baseline.txt current.txt
```

## Privacy & Security

**IMPORTANT**:

- ✅ Test data has PII sanitization applied
- ✅ Test files are in `.gitignore` by default
- ⚠️  Always review sanitization diff before saving
- ⚠️  Don't commit test files unless thoroughly reviewed
- ⚠️  Consider keeping test data local to your machine
