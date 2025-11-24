# Inbox Reaper

Email classification and cleaning system using a hybrid deterministic + AI pipeline with LangGraph orchestration.

## Architecture

### LangGraph-Based Parallel Processing

The system uses **LangGraph** for stateful, parallel email processing:
- **Parallel Processing**: Process multiple emails simultaneously using Send() API
- **State Synchronization**: Thread-safe reducers for concurrent state updates
- **Checkpointing**: Automatic progress persistence with resume capability
- **Graph-Based Workflow**: Declarative DAG for complex orchestration

### Key Components

**State Management** (`langgraph_state.py`):
- `GraphState` - TypedDict with Annotated reducers for thread-safe parallel updates
- Custom reducers: `merge_sender_stats`, `merge_email_headers`
- Built-in reducers: `operator.add` (lists), `operator.or_` (sets)

**Main Graph** (`langgraph_dag.py`):
1. **batch_fetch_headers** - Fetch email headers from IMAP in batch
2. **fan_out_processing** - Spawn parallel subgraphs for each email
3. **aggregate_results** - Collect decisions from parallel subgraphs
4. **batch_fetch_bodies** - Fetch full bodies for AI classification (conditional)
5. **fan_out_ai_classification** - Spawn parallel AI classification tasks
6. **aggregate_ai_results** - Collect AI classification results
7. **batch_delete** - Execute batch IMAP deletion
8. **update_checkpoint** - Persist progress for resumability

**Subgraphs** (`langgraph_subgraphs.py`):
- **email_processing_subgraph** - Deterministic filter pipeline per email
  - check_attachments → check_keywords → check_whitelist → check_sender_pattern
- **ai_classification_subgraph** - AI classification using Ollama

**Checkpoint Management** (`checkpoint_manager.py`):
- UID watermarking for resumable processing
- Progress tracking with ETA calculations
- SQLite-based persistence with LangGraph's SqliteSaver

### Legacy Architecture (dag.py)

The original sequential pipeline is still available:
- Pure functional agents: `ProcessingState -> ProcessingState`
- Sequential execution with optional Google ADK integration
- Use `--no-langgraph` flag to run legacy pipeline (when implemented)

## Installation

```bash
# Install dependencies
uv sync
```

**Dependencies:**
- `langgraph>=0.2.0` - Graph orchestration with checkpointing
- `ollama>=0.1.0` - Local LLM inference
- `pydantic>=2.0.0` - Type-safe state models
- `authlib>=1.3.0` - OAuth 2.0 authentication
- `keyring>=25.0.0` - Secure credential storage
- `click>=8.1.0` - CLI framework

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

### Basic Usage

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

# Increase parallel processing for faster throughput
inbox-reaper process --concurrent-limit 50 --batch-size 200
```

### Checkpoint and Resume

The system automatically creates checkpoints after each batch. If interrupted (CTRL-C, crash, etc.), you can resume:

```bash
# Start processing
inbox-reaper process

# ... interrupted after 50,000 emails processed ...

# Resume from last checkpoint
inbox-reaper process
# Prompts: "Do you want to resume from the last checkpoint? [Y/n]"

# Clear checkpoint and start fresh
inbox-reaper checkpoint clear

# View checkpoint status
inbox-reaper checkpoint status
```

**Checkpoint Features:**
- Exactly-once processing (no duplicate deletions)
- UID watermarking for IMAP pagination
- Persistent sender statistics
- Progress tracking with ETA

### Performance Tuning

**Concurrency Tuning:**
```bash
# For CPU-bound workloads (deterministic filters)
# Rule: concurrent_limit = CPU cores × 2
inbox-reaper process --concurrent-limit 16  # For 8-core CPU

# For GPU-bound workloads (AI classification)
# Rule: concurrent_limit = GPU VRAM / model VRAM
inbox-reaper process --concurrent-limit 6   # For gemma2:2b on 16GB GPU
inbox-reaper process --concurrent-limit 4   # For llama3.2:3b on 16GB GPU
```

**Batch Size Tuning:**
```bash
# Smaller batches: More frequent checkpoints, less memory
inbox-reaper process --batch-size 50

# Larger batches: Fewer IMAP roundtrips, more memory
inbox-reaper process --batch-size 200
```

**Expected Throughput:**
- Deterministic filters only: ~12-50 emails/s (depends on CPU cores)
- With AI classification: ~8-25 emails/s (depends on GPU/model)
- 150,000 emails: ~5-20 hours (vs. 100+ hours sequential)

### CLI Options

**Processing Options:**
- `--model TEXT` - Ollama model name (default: gemma2:2b)
- `--ollama-url TEXT` - Ollama base URL (default: http://localhost:11434)
- `--batch-size INT` - Emails per batch (default: 50)
- `--fetch-size INT` - IMAP fetch size per request (default: 100)
- `--concurrent-limit INT` - Max concurrent parallel tasks (default: 25)
- `--dry-run/--no-dry-run` - Enable dry-run mode (default: True)
- `--checkpoint-path TEXT` - Checkpoint database path (default: checkpoints.db)

**Filter Configuration:**
- `--keywords TEXT` - Critical keywords to trigger KEEP (repeatable)
- `--whitelist-domain TEXT` - Whitelisted domains to trigger KEEP (repeatable)
- `--auto-delete-threshold INT` - Marketing emails before auto-delete (default: 5)
- `--enable-sender-tracking/--no-sender-tracking` - Track sender patterns (default: True)

**Legacy Options:**
- `--use-adk/--no-adk` - Use Google ADK (experimental, default: False)
- `--no-langgraph` - Use legacy sequential pipeline (default: False)

## LangGraph Workflow Example

```python
from inbox_reaper.langgraph_dag import build_langgraph
from inbox_reaper.langgraph_state import create_initial_state
from inbox_reaper.state import Config

# Create configuration
config = Config(
    model_name="gemma2:2b",
    batch_size=50,
    concurrent_ai_limit=25,
    dry_run=True,
)

# Build LangGraph with checkpoint support
graph = build_langgraph(config, checkpoint_path="checkpoints.db")

# Create initial state
initial_state = create_initial_state(config)

# Run graph with automatic checkpointing
final_state = graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": "email_classification_session"}}
)

# Results
print(f"Processed: {final_state['total_processed']}")
print(f"Kept: {final_state['total_kept']}")
print(f"Deleted: {final_state['total_deleted']}")
```

**Parallel Execution Flow:**
1. Fetch email headers (batch of 100)
2. Spawn 100 parallel subgraphs (deterministic filters)
3. Aggregate results → 60 decided, 40 need AI
4. Fetch full bodies for 40 emails
5. Spawn 40 parallel AI classification tasks
6. Aggregate all 100 decisions
7. Batch delete 75 marketing emails
8. Update checkpoint → Resume from here if interrupted

## Design Principles

1. **Parallel-First Architecture** - LangGraph Send() API for dynamic parallelism
2. **Thread-Safe State Management** - Annotated reducers for concurrent updates
3. **Resumable by Default** - Automatic checkpointing with UID watermarking
4. **Layered Decision Pipeline** - Cheap filters first, expensive AI last
5. **Type Safety** - Full type hints with Pydantic validation
6. **Immutability** - State objects are frozen TypedDicts
7. **Pure Functions** - Reducers and agents are pure transformations

## Architecture Documentation

For detailed architecture documentation, see:
- **[DESIGN.md](DESIGN.md)** - High-level design patterns and learnings
  - LangGraph architecture with parallel processing
  - Reducer patterns and thread-safety guarantees
  - Checkpoint format and resume process
  - Performance benchmarks and tuning guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed graph structure
  - Node responsibilities and data flow
  - State schema with reducer annotations
  - Subgraph design patterns
  - Integration points (IMAP, Ollama, SQLite)
- **[docs/PARALLELIZATION.md](docs/PARALLELIZATION.md)** - Parallel processing deep dive
  - Send() API usage patterns
  - Synchronization primitives
  - Batch optimization techniques
  - Performance benchmarks and scaling analysis

## Next Steps

- [x] Add OAuth 2.0 authentication with secure credential storage
- [x] Add LangGraph-based parallel processing architecture
- [x] Add SQLite persistence for progress tracking with checkpointing
- [x] Implement batch database operations with reducers
- [x] Add concurrent AI classification with LangGraph Send() API
- [ ] Add IMAP email fetching using OAuth credentials
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
