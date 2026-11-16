# Inbox Reaper

Cleans an IMAP inbox (Gmail or Outlook) on your Mac. Each email is checked by
deterministic rules, then by a small model trained on your own labels, then,
if still undecided, by a local MLX language model. Nothing leaves your
machine except the IMAP connection itself.

**Precision comes first.** Deleting an important email is unforgivable;
leaving junk behind is harmless. So:

- `process` is a **dry run by default**: it only logs what it would delete.
- Every error keeps the email.
- `learn`, `sample` and `history` are read-only (`BODY.PEEK` and `EXAMINE`:
  nothing is even marked read).
- Every way an email can be deleted is measured on labelled mail with
  `inbox-reaper evaluate` before you trust it.

> **What a delete does:** on Gmail, messages are copied to Trash, then removed
> from INBOX. On Outlook and other providers they are flagged `\Deleted` and
> expunged, so treat a delete as permanent.

## Quick start

Requires macOS on Apple Silicon and Python 3.13 ([uv](https://docs.astral.sh/uv/)).

```bash
uv sync
uv run inbox-reaper download-model          # one-time, ~1.8 GB
uv run inbox-reaper login user@outlook.com  # opens the browser (OAuth)
uv run inbox-reaper process --email user@outlook.com --max-emails 50
```

The last command is a dry run over the 50 oldest emails. Add `--no-dry-run`
only after you've read its log, and ideally after training and evaluating a
combiner (below).

## How an email is decided

`process` streams the INBOX over one IMAP connection, oldest first, fetching
up to 25 messages per request and prefetching the next batch while the current
one is classified. For each email, the first step that decides wins:

1. **Hard keeps:** an important attachment (`.pdf`, `.doc`, `.docx` by
   default), a `--keywords` match, or a `--whitelist-domain` sender. Nothing
   overrides these.
2. **Time rule:** one-time codes, sign-in links and similar are kept for
   their first 30 minutes and deleted once more than an hour old.
3. **Learned rules:** sender or domain rules from `--rules` (see
   [Learned rules](#learned-rules)).
4. **Combiner** (with `--combiner`): a trained model scores the email's
   feature vector. It deletes only when the score clears every keep-labelled
   email it was validated on, keeps when it is clearly safe, and otherwise
   passes the email on with a summary of the signals that fired.
   Without a combiner, only one heuristic runs here: receipts, orders and
   shipping notices are kept. Marketing signals (unsubscribe footers, "view in
   browser") never delete on their own, because bank alerts and statements
   carry them too.
5. **AI:** the local MLX model (Llama 3.2 3B by default) decides whatever is
   left. Waiting emails are grouped into one batched call of up to
   `--concurrent-limit` emails, and the model never runs twice at once.

Deletes are buffered and sent `--batch-size` at a time. There is no resume
state: a re-run classifies whatever is still in the INBOX.

### Features

`features.py` runs independent processors over every email. Each one adds
named values (`<processor>.<name>`, mostly 0/1 flags or scores in [0, 1]):

| Processor | Signals |
| --- | --- |
| `filter` | each deterministic check's verdict (attachment, keyword, whitelist, transactional, marketing) |
| `header` | List-Id, List-Unsubscribe (and one-click), Precedence: bulk, Auto-Submitted, Feedback-ID, bulk-mail platform headers, reply threading, DKIM signer vs From domain, SPF/DMARC, Reply-To elsewhere, recipient count, Outlook bulk/spam scores, GitHub notification reason |
| `link` | link count, share of tracking and off-site links (Outlook Safe Links unwrapped), images, tracking pixels |
| `sender` | free-mail provider, automated address (no-reply, notifications...), subdomain sender, display name, written by a person (the name matches the address), shared "via" a service |
| `content` | record words (receipt, statement, order, reservation...) versus promotional wording, prices, emoji, caps and "!" in the subject; order and booking numbers, account endings, amounts with cents; addressed to you, your name in the subject or greeting, "Dear customer" greetings; word count, link and image density |
| `rule` | a learned rule's keep/delete verdict |
| `time` | lifetime class (minutes / days / forever), expired, fresh, age |
| `history` | from your own mailbox: whether you've read or flagged this email, how often you open this sender's (and domain's) mail, how much they send, whether you've ever written to them, whether it's a thread you took part in |

The `history` features need a mailbox history (see
[Mailbox history](#mailbox-history)). Without one they are all 0.

Text reaches the features and the AI with HTML and leaked CSS stripped,
invisible spacer characters removed, and every URL replaced by
`[link: domain]`.

### The combiner

`combiner.py` is a weighted logistic regression over the feature vector. Its
two thresholds come from cross-validation:

- **Delete threshold:** above the highest score any held-out keep-labelled
  email reached. If a keep scores at the very top, the combiner never deletes.
- **Keep threshold:** low enough that at most 10% of delete-labelled mail
  would be kept.

Keep labels are weighted by how bad losing the email would be (1, 3 or 10).
The metrics `train` prints are nested cross-validation, so they are measured
on emails the model didn't see.

## Training the combiner on your labels

Every file below stays on your machine and holds mailbox data. Keep them
private; `.gitignore` already excludes `samples*.jsonl`, `labels*.jsonl`,
`review*.jsonl` and `history*.json`.

```bash
# 1. A random sample of your inbox (read-only)
inbox-reaper sample --email user@outlook.com -n 300 -o samples.jsonl

# 2. Optional: your mailbox history, for the history features (read-only)
inbox-reaper history --email user@outlook.com -o history.json

# 3. Label it. For each email: keep or delete, how long it stays useful
#    (minutes / days / forever), a category, and how bad losing it would be.
#    Progress is saved after every email; run again to resume.
inbox-reaper label samples.jsonl -o labels-human.jsonl
#    Label first where another labeller (e.g. a model) might be wrong:
inbox-reaper label samples.jsonl -o labels-human.jsonl --prioritize labels-model.jsonl

# 4. Compare labellers per dimension and list keep/delete disagreements
inbox-reaper compare-labels labels-human.jsonl labels-model.jsonl --samples samples.jsonl

# 5. Train; when several files label the same email, the first wins
inbox-reaper train samples.jsonl --labels labels-human.jsonl --labels labels-model.jsonl \
  --history history.json

# 6. Wrong deletes, precision and recall for each deletion path:
#    today's pipeline versus the pipeline with the combiner
inbox-reaper evaluate samples.jsonl --labels labels-human.jsonl \
  --combiner combiner.json --history history.json

# 7. Use it (add --history history.json if you trained with it)
inbox-reaper process --email user@outlook.com --combiner combiner.json
```

Training applies each label's lifetime. An email labelled *keep* but useful
for only minutes or days counts as *delete* if it was already past that when
sampled.

With a few hundred labels the delete threshold is set by a handful of
borderline keeps, so results move between training runs. Read the wrong
deletes that `evaluate` lists, not just the totals.

### Mailbox history

`inbox-reaper history` scans the INBOX (sender, date, read and flagged state of
every message) and the Sent folder (who you've written to, and the Message-IDs
of what you sent), read-only. The first scan of a large mailbox is slow (about
15 minutes for 77,000 messages on Outlook). After that the file is updated in
place: only new messages are fetched, and read and flagged state comes from
two cheap `UID SEARCH` commands, so an update takes seconds. If the server
renumbers a folder (a new UIDVALIDITY), that folder is rescanned; `--full`
forces a rescan.

A combiner trained with `--history` remembers it. Run it with
`process --history history.json`, which updates the file before classifying.
`evaluate`, and `process` without a history, refuse to run such a combiner,
so the history features can never silently drop to 0.

Whether history helps depends on your mailbox: compare `train` with and
without `--history` on the same labels. If you keep everything marked read,
the read-state signals carry no information.

## Learned rules

`learn` classifies your inbox read-only and writes sender and domain rules you
can review and edit. `process --rules` applies them after the hard keeps
(which a rule can never override) and before the combiner and AI.

```bash
inbox-reaper learn --email user@outlook.com -o rules.json
inbox-reaper learn --email user@outlook.com --no-ai   # fast, deterministic signals only
inbox-reaper process --email user@outlook.com --rules rules.json
```

A rule is learned only when a sender (or domain) has at least `--min-support`
emails (default 3) and `--min-agreement` of them agree (default 90%). Any email
kept by an attachment, keyword or whitelist check blocks a delete rule for its
sender. Domain rules need two or more senders, are never learned for free-mail
domains such as gmail.com, and are dropped if a sender rule on that domain
disagrees. Sender rules take precedence over domain rules, and domain rules
also cover subdomains:

```json
{"match": "domain", "value": "e.shop.com", "action": "delete",
 "support": 42, "agreement": 1.0, "source": "learned"}
```

Delete an entry to drop it, flip its `action`, or add your own with
`"source": "manual"`.

## Configuration

Options can come from a YAML or JSON file. Command-line flags override it.

```bash
cp config.example.yaml config.yaml
inbox-reaper process --config config.yaml
```

```yaml
email: user@outlook.com
model_name: mlx-community/Llama-3.2-3B-Instruct-4bit
concurrent_ai_limit: 8        # emails per batched AI call
batch_size: 50                # UIDs per delete
max_emails: null              # oldest N only; null = all
keywords: [important, urgent]
whitelist_domains: [example.com]
important_extensions: [.pdf, .doc, .docx]
rules_path: null              # from `learn`
combiner_path: null           # from `train`
history_path: null            # from `history`, if the combiner uses it
dry_run: true
```

See `config.example.yaml` for every option.

## Commands

| Command | What it does |
| --- | --- |
| `login EMAIL [--provider]` | OAuth sign-in; the token is kept in the system keychain |
| `logout EMAIL`, `accounts` | Remove stored credentials; list accounts |
| `test EMAIL` | Check the IMAP connection |
| `diagnose EMAIL [--refresh]` | Inspect the stored token, optionally test a refresh |
| `download-model [--model]` | Download the MLX model |
| `process` | Classify the INBOX and delete (dry run by default) |
| `learn` | Read-only: classify and write sender/domain rules |
| `sample` | Read-only: save a random sample of emails for labelling |
| `history` | Read-only: save or update INBOX and Sent behaviour for the history features |
| `label` | Label sampled emails interactively, along several dimensions |
| `compare-labels` | Agreement between labellers, and their disagreements |
| `train` | Train the combiner from labelled samples |
| `evaluate` | Wrong deletes, precision and recall per deletion path |
| `fetch-email --email --uid` | Save one message as `.eml` (debugging) |
| `version` | Print the version |

`process` options:

| Option | Default | |
| --- | --- | --- |
| `--config PATH` | | YAML or JSON configuration file |
| `--email TEXT` | | Account to process |
| `--dry-run/--no-dry-run` | dry run | Log deletes instead of performing them |
| `--max-emails INT` | all | Process only the oldest N |
| `--model TEXT` | Llama 3.2 3B 4-bit | MLX model (Hugging Face ID) |
| `--concurrent-limit INT` | 8 | Emails per batched AI call |
| `--batch-size INT` | 50 | UIDs per delete |
| `--keywords TEXT` | | Keep emails containing this (repeatable) |
| `--whitelist-domain TEXT` | | Keep emails from this domain (repeatable) |
| `--rules FILE` | | Rules from `learn` |
| `--combiner FILE` | | Combiner from `train` |
| `--history FILE` | | Mailbox history for a combiner trained with one (created if missing, then updated) |

## Performance

The AI dominates run time; everything decided before it is nearly free.

- **Decide more before the model:** a combiner, `--rules`, keywords and
  whitelisted domains all mean fewer emails reach the AI.
- **Batching:** `--concurrent-limit` makes each AI email about 15-25% cheaper
  on an M-series Mac. Past about 16 emails per batch it gets slower per email.
- **A smaller model** (`mlx-community/Llama-3.2-1B-Instruct-4bit`) is faster but
  less accurate.
- **Memory:** the 3B model needs about 2-3 GB of unified memory. On a 16-18 GB
  Mac, don't run two model processes at once, or the machine will swap.

## Python API

```python
import asyncio

from inbox_reaper.langgraph_streaming import run_learning_pipeline, run_streaming_pipeline
from inbox_reaper.state import Config

config = Config(email="user@outlook.com", dry_run=True)

# Read-only: classify the inbox and learn sender/domain rules
rules = asyncio.run(run_learning_pipeline(config, use_ai=False))
rules.save("rules.json")

# Apply them (dry_run=True only logs what would be deleted)
asyncio.run(run_streaming_pipeline(config.model_copy(update={"rules_path": "rules.json"})))
```

## Code map

| Module | Role |
| --- | --- |
| `cli.py` | Click commands |
| `langgraph_streaming.py` | IMAP session, fetch and prefetch, decision order, `process` / `learn` / `sample` runners, AI batching |
| `imap_client.py` | Async IMAP client: XOAUTH2, batched fetch, MIME parsing, read-only header fetch, delete |
| `text_utils.py` | HTML/CSS stripping, URL compaction, link and image extraction |
| `agents.py` | Deterministic filters and the AI classifier |
| `features.py` | Feature processors and the time rule |
| `history.py` | Read-only mailbox history (INBOX and Sent) |
| `combiner.py` | Logistic-regression combiner and precision-first thresholds |
| `labeling.py` | Samples, labels, comparison, training, evaluation |
| `rules.py` | Learned rules: load, save, match, learn |
| `mlx_backend.py` | MLX model loading and (batched) generation |
| `batch_coordinator.py` | Buffers delete UIDs and flushes them in batches |
| `oauth_flow.py`, `oauth_config.py`, `credential_helper.py` | OAuth with PKCE (Thunderbird's public client IDs), token refresh, keychain storage |
| `config_loader.py`, `state.py` | Configuration loading; Pydantic models |

## Authentication

`login` uses OAuth 2.0 with PKCE and Thunderbird's public client IDs for
Gmail and Outlook, so you don't need to register an app. Tokens are stored
with [keyring](https://pypi.org/project/keyring/) (Keychain on macOS) and
refreshed automatically when they're within 5 minutes of expiry.

## Development

```bash
uv sync
uv run setup-dev                    # dev dependencies and pre-commit hooks
uv run pre-commit run --all-files   # ruff and mypy, the same checks as CI
uv run pytest
```

## License

MIT
