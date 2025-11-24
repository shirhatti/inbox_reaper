# Migrate to aioimaplib for better async IMAP support

## Problem
Currently using the standard library `imaplib` which has several limitations:
- **Synchronous only**: Requires `asyncio.to_thread()` wrappers, adding overhead
- **Poor error handling**: Only raises generic `IMAP4.error` with string messages, no structured error codes
- **No built-in OAuth refresh**: Manual string parsing required to detect auth failures
- **Low-level API**: Requires manual handling of many IMAP protocol details

## Proposed Solution
Migrate to `aioimaplib` which provides:
- **Native async/await support**: Eliminates need for thread pool wrappers
- **Better error handling**: More structured exceptions
- **Modern API**: Cleaner interface for common operations
- **Active maintenance**: Well-maintained library with good OAuth2 support

## Benefits
1. **Performance**: Native async eliminates thread overhead
2. **Reliability**: Better error detection and handling
3. **Maintainability**: Cleaner code, less manual protocol handling
4. **Consistency**: Matches our async-first architecture (LangGraph streaming)

## Migration Scope
Files to update:
- `src/inbox_reaper/imap_client.py` - Main IMAP client implementation
- `src/inbox_reaper/langgraph_streaming.py` - Remove `asyncio.to_thread()` wrappers
- `tests/test_streaming.py` - Update mocks for new library
- `pyproject.toml` - Add `aioimaplib` dependency

## Acceptance Criteria
- [ ] All IMAP operations use native async (no `asyncio.to_thread()`)
- [ ] OAuth token refresh works automatically on auth failures
- [ ] All existing tests pass
- [ ] Error handling is more robust (no string parsing for error detection)
- [ ] Performance is equal or better than current implementation

## References
- aioimaplib: https://github.com/bamthomas/aioimaplib
- Current issue: Auth failure detection relies on fragile string matching
