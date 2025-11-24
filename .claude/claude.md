# Claude Code Instructions for inbox_reaper

## Pre-commit Hooks - REQUIRED

**CRITICAL**: This project uses pre-commit hooks to maintain code quality and ensure CI passes.

### On First Setup or When Starting Work

ALWAYS run this command before making any code changes:

```bash
uv run setup-dev
```

This will:
- Install all development dependencies
- Install pre-commit hooks automatically
- Ensure hooks are up to date

### Before Every Commit

Pre-commit hooks will run automatically on `git commit`. However, if you want to run them manually on all files:

```bash
uv run pre-commit run --all-files
```

### Why This Matters

- Pre-commit hooks run ruff, black, and mypy locally
- These are the SAME checks that run in CI
- If pre-commit passes, CI will pass
- Prevents wasted time with CI failures

### If You Skip This

If pre-commit hooks are not installed:
1. Code may not pass CI checks
2. You'll need to make additional commits to fix formatting
3. Wastes time and creates messy git history

### Verification

To check if pre-commit is available:

```bash
uv run pre-commit --version
```

To check if hooks are installed in this repo:

```bash
ls -la .git/hooks/pre-commit
```

You should see the pre-commit hook file present.
