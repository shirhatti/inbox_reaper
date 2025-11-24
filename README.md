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
uv pip install -e .
```

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

## Next Steps

- [ ] Add IMAP email fetching
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
uv pip install -e ".[dev]"
uv run setup-dev
```

This will:
- Install all development dependencies
- Set up pre-commit hooks (runs ruff, black, mypy automatically)
- Verify your environment is ready

### Manual Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks manually
pre-commit install
```

### Code Quality

**Pre-commit hooks run automatically on every commit!** They check:
- Ruff linting and formatting
- Black formatting
- Mypy type checking

```bash
# Run pre-commit hooks manually on all files
pre-commit run --all-files

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
