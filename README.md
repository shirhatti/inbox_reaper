# Inbox Reaper

Email classification and cleaning system using a hybrid deterministic + AI pipeline.

## Architecture

### State-Based Design

All application state is encapsulated in immutable Pydantic models (`state.py`):
- `ProcessingState` - Single source of truth flowing through the pipeline
- `Email` - Immutable email data
- `EmailDecision` - Classification decision with metadata
- `Config` - Pipeline configuration

### Pure Functional Agents

Each agent is a pure function: `ProcessingState -> ProcessingState` (`agents.py`):
1. **attachment_filter_agent** - Check for important attachments
2. **keyword_filter_agent** - Check for critical keywords
3. **whitelist_filter_agent** - Check against whitelisted domains
4. **sender_pattern_filter_agent** - Check sender history patterns
5. **ai_classifier_agent** - AI classification using Ollama (only non-pure)
6. **log_progress_agent** - Display progress
7. **check_termination_agent** - Determine if processing should stop

### DAG Workflow

The agent pipeline is defined in `dag.py`:
- Sequential execution for now
- Easily extensible to parallel/conditional flows
- Google ADK integration scaffolded (placeholder)

## Installation

```bash
# Install dependencies
uv sync
```

## Authentication

Inbox Reaper uses OAuth 2.0 for secure email access. Credentials are stored securely using the [keyring](https://pypi.org/project/keyring/) library, which automatically uses your system's native credential storage:
- **macOS**: Keychain
- **Windows**: Windows Credential Locker
- **Linux**: Secret Service (GNOME Keyring / KWallet)

### Login to Email Account

```bash
# Gmail (auto-detected)
inbox-reaper login user@gmail.com

# Outlook/Hotmail (auto-detected)
inbox-reaper login user@outlook.com

# Specify provider explicitly
inbox-reaper login user@company.com --provider gmail
```

This will:
1. Open your browser for OAuth authentication
2. Securely store your credentials
3. Automatically refresh tokens when needed

### Manage Accounts

```bash
# List all stored accounts
inbox-reaper accounts

# List with detailed information
inbox-reaper accounts --verbose

# Test IMAP connection
inbox-reaper test user@gmail.com

# Remove stored credentials
inbox-reaper logout user@gmail.com
```

### OAuth Details

Inbox Reaper uses **Thunderbird's public OAuth client IDs** for Gmail and Outlook with **PKCE** (Proof Key for Code Exchange) for enhanced security:
- ✅ No need to create your own OAuth app
- ✅ Works out of the box
- ✅ PKCE protection against authorization code interception
- ✅ Secure and privacy-focused
- ⚠️ Credentials are stored locally on your machine only

**Technical Implementation:**
- Uses [Authlib](https://docs.authlib.org/) for OAuth 2.0 with PKCE
- SHA256 code challenge method for maximum security
- Automatic token refresh when expired

## Usage

### Basic Usage (Mock Emails)

```bash
# Run with defaults (dry-run mode, gemma2:2b model)
inbox-reaper process

# Customize model and settings
inbox-reaper process --model llama3.2:3b --batch-size 100

# Add keywords and whitelisted domains
inbox-reaper process \
  --keywords "important" \
  --keywords "Mario Romo" \
  --whitelist-domain "gmail.com" \
  --whitelist-domain "wellsfargo.com"
```

### Options

- `--model TEXT` - Ollama model name (default: gemma2:2b)
- `--ollama-url TEXT` - Ollama base URL (default: http://localhost:11434)
- `--batch-size INT` - Emails per batch (default: 50)
- `--concurrent-limit INT` - Max concurrent AI requests (default: 25)
- `--dry-run/--no-dry-run` - Enable dry-run mode (default: True)
- `--use-adk/--no-adk` - Use Google ADK (experimental, default: False)
- `--keywords TEXT` - Critical keywords (repeatable)
- `--whitelist-domain TEXT` - Whitelisted domains (repeatable)

## State Flow Example

```python
# Initial state
state = ProcessingState(
    config=config,
    emails=[email1, email2, ...],
    decisions=[],
    processed_uids=set(),
)

# Flow through pipeline
state = attachment_filter_agent(state)  # May add some decisions
state = keyword_filter_agent(state)     # May add more decisions
state = whitelist_filter_agent(state)   # May add more decisions
state = sender_pattern_filter_agent(state)  # May add more decisions
state = ai_classifier_agent(state)      # Classify remaining emails
state = log_progress_agent(state)       # Display progress
state = check_termination_agent(state)  # Check if should stop

# Final state has all decisions
print(f"Processed: {state.total_processed}")
print(f"Kept: {state.total_kept}")
print(f"Deleted: {state.total_deleted}")
```

## Design Principles

1. **Immutability** - All state objects are frozen Pydantic models
2. **Pure Functions** - Agents are pure transformations (except AI calls)
3. **Single Source of Truth** - ProcessingState contains everything
4. **Layered Decision Pipeline** - Cheap filters first, expensive AI last
5. **Type Safety** - Full type hints with Pydantic validation

## Testing & Benchmarking

### Golden Dataset Testing

Create a test dataset from your real emails with automatic PII sanitization:

```bash
# Fetch and label 20 emails interactively
inbox-reaper create-test-data user@gmail.com --count 20
```

This will:
1. Fetch emails via IMAP
2. Auto-sanitize PII (emails, phone numbers, credit cards, etc.)
3. Show diff in $EDITOR for review
4. Prompt you to label each email (keep/delete)
5. Save sanitized test cases to `test_data/golden/`

Run golden dataset tests:

```bash
# Run classifier against labeled test data
pytest tests/test_golden_dataset.py --run-golden

# View detailed metrics
pytest tests/test_golden_dataset.py --run-golden -v
```

**Note**: Golden tests are skipped by default in CI. See `test_data/README.md` for details.

## Next Steps

- [x] Add OAuth 2.0 authentication with secure credential storage
- [x] Add IMAP email fetching using OAuth credentials
- [x] Add golden dataset testing framework
- [ ] Add SQLite persistence for progress tracking
- [ ] Implement batch database operations
- [ ] Add concurrent AI classification (asyncio + semaphore)
- [ ] Add pipelined I/O (fetch while processing)
- [ ] Implement full Google ADK integration
- [ ] Add email deletion functionality

## Development

### Quick Setup (Recommended)

```bash
# Install dependencies and set up pre-commit hooks automatically
uv sync --extra dev
uv run setup-dev
```

This will:
- Install all development dependencies
- Set up pre-commit hooks (runs ruff, black, mypy automatically)
- Verify your environment is ready

### Manual Setup

```bash
# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks manually
uv run pre-commit install
```

### Code Quality

**Pre-commit hooks run automatically on every commit!** They check:
- Ruff linting and formatting
- Black formatting
- Mypy type checking

```bash
# Run pre-commit hooks manually on all files
uv run pre-commit run --all-files

# Individual tool commands (optional)
ruff check src/          # Lint
ruff format src/         # Format
black src/               # Additional formatting
mypy src/                # Type check
pytest                   # Run tests
```

**Important**: If pre-commit hooks pass locally, CI will pass. Always run `setup-dev` to ensure hooks are installed.

## License

MIT
