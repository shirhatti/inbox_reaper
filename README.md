# Inbox Reaper

Email classification and cleaning system: an IMAP inbox is streamed through
deterministic filters, a feature-vector combiner trained on your own labels,
and a local MLX language model; messages classified as deletable are deleted in
batches. Precision comes first: a wrong delete is the error that matters, so
every deletion path can be measured on labelled mail (`inbox-reaper evaluate`).

## Architecture

### How `process` works

`inbox-reaper process` runs the streaming pipeline in `langgraph_streaming.py`:

1. **Connect** - one shared IMAP connection (`IMAPSession`), authenticated with
   the stored OAuth token, which is refreshed first if it expires within 5
   minutes. Commands are serialized on that connection, and a dropped
   connection is reconnected once.
2. **Search** - `UID SEARCH ALL` on the INBOX, oldest first (`--max-emails`
   takes the oldest N).
3. **Fetch** - full messages in batches of up to 25 per `UID FETCH`. The next
   batch is prefetched in the background while the current one is classified,
   so network latency overlaps with model inference.
4. **Classify** - each email goes through a one-node LangGraph graph
   (`process_email`). Up to 2 x `--concurrent-limit` emails are in flight at
   once, so filter decisions never wait behind the model; the first step that
   returns a decision wins:
   1. *Hard keeps* - important attachments, `--keywords`, `--whitelist-domain`
      (always first, never overridden)
   2. *Time rule* - one-time codes and sign-in links are kept for their first
      30 minutes and deleted once over an hour old
   3. *Learned rules* - sender/domain rules from `--rules` (see below)
   4. With `--combiner`: the *feature vector* (below) is scored by the trained
      combiner, which deletes or keeps only when confident.
      Without one: the *heuristics* - transactional patterns keep. Marketing
      indicators (unsubscribe text, "view in browser", tracking subdomains)
      never delete on their own - bank alerts and statements carry them too -
      so those emails go to the AI
   5. *AI* - the MLX model, for whatever is still undecided (with a combiner it
      also receives a one-line summary of the signals that fired). A single
      worker groups waiting emails into one batched generation call of up to
      `--concurrent-limit` emails, in a thread so fetching continues; the model
      is never called concurrently

   Any error while classifying an email keeps it.
5. **Delete** - delete decisions are buffered by `BatchCoordinator` and sent as
   one IMAP delete per `--batch-size` UIDs (logged only under `--dry-run`, the
   default), with a final flush at the end.

`inbox-reaper learn` reuses steps 1-4 read-only (it has no delete path) and
turns the decisions into a reviewable `rules.json`.

### Feature vector and combiner

`features.py` runs independent processors over each email; each appends named
features (`<processor>.<name>`, mostly 0/1 flags or scores in [0, 1]):

| Processor | Signals |
| --- | --- |
| `filter` | each deterministic check's verdict (attachment, keyword, whitelist, transactional, marketing) |
| `header` | List-Id, List-Unsubscribe (and one-click), Precedence: bulk, Auto-Submitted, Feedback-ID, bulk-mail platform headers, reply threading, DKIM signer vs From domain, SPF/DMARC, Reply-To elsewhere, recipient count, Outlook bulk/spam scores, GitHub notification reason |
| `link` | link count, share of tracking and off-site links (Outlook Safe Links unwrapped), images, tracking pixels |
| `sender` | free-mail provider, automated local part (no-reply, notifications, ...), subdomain sender, display name |
| `rule` | a learned rule's keep/delete verdict |
| `time` | lifetime class (minutes / days / forever), expired, fresh, age |

`combiner.py` is a logistic regression over that vector, trained on labelled
samples. It deletes only above a threshold set from cross-validation so that
no held-out keep-labelled email would have been deleted, keeps below a low
threshold, and sends everything in between to the AI. Keep labels are
weighted by how bad losing the email would be.

Email bodies reach the models with HTML stripped, invisible spacer characters
removed and every URL replaced by `[link: domain]` (URLs were ~30% of the text
on a real inbox sample).

### Key Modules

| Module | Role |
| --- | --- |
| `cli.py` | Click commands: `login`, `process`, `learn`, `test`, ... |
| `langgraph_streaming.py` | IMAP session, producer and prefetch, classification order, `process`/`learn` runners |
| `imap_client.py` | Async IMAP client (aioimaplib): XOAUTH2 login, batched fetch, MIME parsing, delete |
| `agents.py` | Deterministic filters and the AI classifier (`classify_with_ai`) |
| `features.py` | Feature processors, time-sensitivity rule, signal summaries for the AI |
| `combiner.py` | Trained logistic-regression combiner and precision-first thresholds |
| `labeling.py` | Samples, multi-dimensional labels, comparison, training, evaluation |
| `rules.py` | `RuleSet` (load/save/match) and `RuleLearner` |
| `mlx_backend.py` | MLX model loading and text generation |
| `batch_coordinator.py` | Buffers UIDs and flushes them in batches |
| `oauth_flow.py`, `credential_helper.py` | OAuth login and refresh, keyring storage |
| `state.py` | Pydantic models: `Config`, `Email`, `EmailDecision` |

## Installation

```bash
# Install dependencies
uv sync
```

**Dependencies:**
- `langgraph>=1.2` - Graph orchestration
- `mlx-lm>=0.32` - Local LLM inference optimized for Apple Silicon
- `pydantic>=2.14` - Type-safe state models
- `authlib>=1.9` - OAuth 2.0 authentication
- `keyring>=25.7` - Secure credential storage
- `click>=8.5` - CLI framework

**Note**: This tool is optimized for **macOS with Apple Silicon** (M1/M2/M3/M4/M5). MLX provides native GPU acceleration on Apple hardware.

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
# First-time setup: Download model (one-time, ~1.8GB)
inbox-reaper download-model

# Run with defaults (dry-run mode, Llama-3.2-3B-Instruct-4bit)
inbox-reaper process --email user@gmail.com

# Customize model and settings
inbox-reaper process --email user@gmail.com \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --batch-size 100

# Add keywords and whitelisted domains
inbox-reaper process --email user@gmail.com \
  --keywords "important" \
  --keywords "Mario Romo" \
  --whitelist-domain "gmail.com" \
  --whitelist-domain "wellsfargo.com"
```

### Learned Rules (review before deleting)

`learn` classifies your inbox **read-only** and writes deterministic sender/domain
rules you can review and edit. `process --rules` then applies them after the
attachment/keyword/whitelist checks (which a rule can never override) and before
heuristics and AI.

```bash
# Read-only: never deletes, regardless of --dry-run
inbox-reaper learn --email user@gmail.com -o rules.json
inbox-reaper learn --email user@gmail.com --no-ai   # fast, deterministic signals only

# Review/edit rules.json, then apply
inbox-reaper process --email user@gmail.com --rules rules.json
```

A rule is learned only when a sender (or domain) has at least `--min-support`
emails (default 3) and `--min-agreement` of them agree (default 90%). Any email
kept by an attachment, keyword or whitelist check blocks a delete rule for that
sender. Domain rules need two or more senders, are never learned for free-mail
domains such as gmail.com, and are dropped if a sender rule on that domain
disagrees. Each entry looks like:

```json
{"match": "domain", "value": "e.shop.com", "action": "delete",
 "support": 42, "agreement": 1.0, "source": "learned"}
```

Delete an entry to drop it, flip `action`, or add your own with `"source": "manual"`.
Sender rules take precedence over domain rules; domain rules also cover subdomains.

### Training the combiner on your own labels

All files stay on your machine; samples contain email contents, so keep them
private.

```bash
# 1. Read-only random sample of your inbox
inbox-reaper sample --email user@gmail.com -n 300 -o samples.jsonl

# 2. Label it: for each email, keep/delete, how long it stays useful
#    (minutes/days/forever), a category, and how bad losing it would be.
#    Resumable; other labellers' answers are never shown.
inbox-reaper label samples.jsonl -o labels-human.jsonl
#    Label where other labellers (e.g. a model) disagree first:
inbox-reaper label samples.jsonl -o labels-human.jsonl --prioritize labels-model.jsonl

# 3. Compare labellers per dimension and list keep/delete disagreements
inbox-reaper compare-labels labels-human.jsonl labels-model.jsonl --samples samples.jsonl

# 4. Train (when several files label an email, the first wins)
inbox-reaper train samples.jsonl --labels labels-human.jsonl --labels labels-model.jsonl

# 5. Measure delete precision and recall per path, today vs with the combiner
inbox-reaper evaluate samples.jsonl --labels labels-human.jsonl --combiner combiner.json

# 6. Use it
inbox-reaper process --email user@gmail.com --combiner combiner.json
```

Training applies each label's lifetime: an email labelled *keep* but useful for
only minutes or days counts as *delete* once it is past that when sampled.

### Configuration Files

Instead of passing CLI arguments each time, you can use a configuration file:

```bash
# Copy example config
cp config.example.yaml config.yaml
# Or: cp config.example.json config.json

# Edit your config file
vim config.yaml

# Run with config file (CLI args override file values)
inbox-reaper process --config config.yaml

# Mix config file with CLI overrides
inbox-reaper process --config config.yaml --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

**Example config.yaml:**
```yaml
# AI Settings
model_name: mlx-community/Llama-3.2-3B-Instruct-4bit
concurrent_ai_limit: 25

# Email
email: user@gmail.com

# Filters
keywords:
  - important
  - urgent
whitelist_domains:
  - example.com

# Safety
dry_run: true
```

See `config.example.yaml` or `config.example.json` for all available options.

### Interrupting and Re-running

`inbox-reaper process` keeps no resume state: every run searches the whole
INBOX and classifies each message it finds. Deletions are applied per batch, so
if a run is interrupted (CTRL-C, crash, lost connection), emails from batches
that were already deleted are gone and the next run simply works through what
remains. Kept emails, and everything in a `--dry-run`, are classified again.

To skip the AI on a re-run, run `inbox-reaper learn` once and pass the result
with `--rules rules.json`; senders it covers are then decided without the model.

### Performance Tuning

AI inference dominates the run time. Batching (`--concurrent-limit`) makes
each AI email about 15-25% cheaper on an M-series Mac with the default 3B
model; batches past ~16 get slower per email, so the default is 8. The
biggest win is still letting more emails be decided before the model:

- Run `inbox-reaper learn` (or `learn --no-ai` for a quick pass) and pass
  `--rules rules.json`; senders covered by a rule skip the model entirely.
- Add `--keywords` and `--whitelist-domain` entries for mail you always keep.
- A smaller model (for example `mlx-community/Llama-3.2-1B-Instruct-4bit`) is
  faster at some cost in accuracy.

**Batch Size:**
```bash
# Smaller batches: Deletions applied more often
inbox-reaper process --batch-size 50

# Larger batches: Fewer IMAP delete round trips
inbox-reaper process --batch-size 200
```

### CLI Options

**Configuration:**
- `--config PATH` - Path to configuration file (YAML or JSON)

**Processing Options:**
- `--model TEXT` - MLX model name from Hugging Face (default: mlx-community/Llama-3.2-3B-Instruct-4bit)
- `--batch-size INT` - Emails per batch (default: 50)
- `--concurrent-limit INT` - Emails per batched AI call (default: 8)
- `--dry-run/--no-dry-run` - Enable dry-run mode (default: True)
- `--email TEXT` - Email address to process (required unless in config file)
- `--max-emails INT` - Stop after this many emails (for testing)

**Filter Configuration:**
- `--keywords TEXT` - Critical keywords to trigger KEEP (repeatable)
- `--whitelist-domain TEXT` - Whitelisted domains to trigger KEEP (repeatable)
- `--rules FILE` - Learned rules from `inbox-reaper learn`, applied before heuristics and AI
- `--combiner FILE` - Trained combiner from `inbox-reaper train`; replaces the heuristics as decision-makers (they become features) and sends only contested emails to the AI

## Python API

```python
import asyncio

from inbox_reaper.langgraph_streaming import run_learning_pipeline, run_streaming_pipeline
from inbox_reaper.state import Config

config = Config(email="user@gmail.com", dry_run=True)

# Read-only: classify the inbox and learn sender/domain rules
rules = asyncio.run(run_learning_pipeline(config, use_ai=False))
rules.save("rules.json")

# Apply them (dry_run=True only logs what would be deleted)
asyncio.run(run_streaming_pipeline(config.model_copy(update={"rules_path": "rules.json"})))
```

## Design Principles

1. **Precision over recall** - a wrong delete is unforgivable, a missed one is
   harmless: dry-run by default, hard keeps can't be overridden, errors keep the
   email, `learn`/`sample` are read-only, and the combiner deletes only above
   every held-out keep
2. **Layered decision pipeline** - cheap deterministic checks first, the model last
3. **Reviewable rules** - learned decisions are plain JSON you can edit before applying
4. **Measured, not assumed** - `evaluate` reports precision per deletion path on your labels
5. **Overlap I/O with inference** - one IMAP connection, batched fetches, background prefetch
6. **Type safety** - full type hints with Pydantic models, checked by mypy

## Next Steps

- [x] OAuth 2.0 authentication with secure credential storage
- [x] IMAP email fetching using OAuth credentials
- [x] Pipelined I/O (fetch while classifying)
- [x] Batched email deletion
- [x] Read-only rule learning (`learn`) and rule application (`process --rules`)
- [x] Batched AI classification (`--concurrent-limit`)
- [x] Feature-vector combiner trained on your labels, with per-path precision evaluation
- [ ] Resume an interrupted run without re-classifying kept emails

## Development

### Quick Setup (Recommended)

```bash
# Install dependencies and set up pre-commit hooks automatically
uv sync
uv run setup-dev
```

This will:
- Install all development dependencies
- Set up pre-commit hooks (runs ruff and mypy automatically)
- Verify your environment is ready

### Manual Setup

```bash
# Install development dependencies
uv sync

# Install pre-commit hooks manually
uv run pre-commit install
```

### Code Quality

**Pre-commit hooks run automatically on every commit!** They check:
- Ruff linting and formatting
- Mypy type checking

```bash
# Run pre-commit hooks manually on all files
uv run pre-commit run --all-files

# Individual tool commands (optional)
ruff check src/          # Lint
ruff format src/         # Format
mypy src/                # Type check
pytest                   # Run tests
```

**Important**: If pre-commit hooks pass locally, CI will pass. Always run `setup-dev` to ensure hooks are installed.

## License

MIT
