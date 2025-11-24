# Streaming Architecture with Backpressure

## Problem Statement

The current LangGraph implementation has a fundamental scalability issue:
1. **Batch-all approach**: `batch_fetch_headers` tries to fetch ALL headers upfront
2. **Memory explosion**: Can't hold 150K email headers in memory
3. **No streaming**: Everything is batch-oriented, not streaming
4. **No backpressure**: Can't signal upstream to slow down

## Requirements

Following .NET's Bounded Channels pattern, we need:
1. **Producer-Consumer Pattern**: Headers producer, email processors as consumers
2. **Bounded Queue**: Limit in-flight items (e.g., 1000 emails max in memory)
3. **Backpressure**: When queue is full, producer pauses
4. **Dynamic Refilling**: As queue empties, producer resumes fetching
5. **Deadlock Prevention**: Avoid blocking on multiple batching operations

## Proposed Architecture

### High-Level Flow

```
IMAP Producer (async loop)
    ↓ (bounded channel, size=1000)
Email Worker Pool (parallel consumers)
    ↓ (collect decisions)
Batch Coordinator (batch delete when ready)
```

### LangGraph Implementation

```python
# State with bounded queue concept
class StreamingState(TypedDict):
    # Producer state
    current_imap_offset: str | None  # Last fetched UID
    producer_active: bool  # Is producer still fetching?

    # Bounded queue (implemented via state + semaphore)
    in_flight_uids: Annotated[set[str], operator.or_]  # Currently processing
    queue_size: int  # Current queue size
    max_queue_size: int  # Backpressure threshold

    # Worker state
    email_headers: Annotated[dict[str, dict], merge_headers]  # Buffered headers
    decisions: Annotated[list[dict], operator.add]
    sender_stats: Annotated[dict[str, dict], merge_sender_stats]

    # Batch coordinator
    pending_deletes: Annotated[list[str], operator.add]
    delete_batch_size: int  # Trigger delete when this many accumulated
```

### Graph Structure

```
START
  ↓
fetch_batch_node (produces batch of N headers)
  ↓
check_backpressure
  ├─ queue_full? → sleep_node → check_backpressure
  └─ queue_available? → fan_out_to_workers
                          ↓
                     [parallel workers]
                     process_email_node
                          ↓
                     mark_complete_node (remove from in_flight)
                          ↓
                     accumulate_decisions
                          ↓
                     check_batch_delete
                          ├─ batch_ready? → execute_delete
                          └─ not_ready? → continue
  ↓
check_continuation
  ├─ more_emails? → fetch_batch_node (loop)
  └─ done? → final_delete → END
```

### Key Components

#### 1. Backpressure Control

```python
def check_backpressure(state: StreamingState) -> Literal["wait", "proceed"]:
    \"\"\"Check if queue has capacity.\"\"\"
    in_flight = len(state["in_flight_uids"])
    max_size = state["max_queue_size"]

    if in_flight >= max_size:
        return "wait"  # Producer pauses
    elif in_flight < max_size * 0.3:  # Refill when <30% full
        return "proceed"  # Producer resumes
    else:
        return "proceed"  # Continue normal operation
```

#### 2. Streaming Fetch

```python
def fetch_batch_node(state: StreamingState) -> StreamingState:
    \"\"\"Fetch next batch with IMAP windowing.\"\"\"
    offset = state["current_imap_offset"]
    batch_size = 100  # Fetch 100 at a time

    # IMAP: SEARCH UID <last_uid+1:*> (window fetch)
    new_headers = imap_client.fetch_headers(
        start_uid=offset,
        limit=batch_size
    )

    if not new_headers:
        return {**state, "producer_active": False}

    # Update offset for next fetch
    last_uid = max(new_headers.keys())

    return {
        **state,
        "email_headers": new_headers,
        "in_flight_uids": set(new_headers.keys()),
        "current_imap_offset": last_uid,
        "queue_size": len(state["in_flight_uids"]) + len(new_headers)
    }
```

#### 3. Worker Node

```python
def process_email_node(state: StreamingState) -> StreamingState:
    \"\"\"Process single email (runs in parallel).\"\"\"
    email_uid = state.get("current_email_uid")
    email = state["email_headers"][email_uid]

    # Run deterministic filters
    decision = run_filters(email, state)

    # Mark as complete (remove from in_flight)
    return {
        **state,
        "decisions": [decision],
        "in_flight_uids": {email_uid},  # Reducer removes this
        "pending_deletes": [email_uid] if decision == "delete" else []
    }
```

#### 4. Batch Delete Coordinator

```python
def check_batch_delete(state: StreamingState) -> Literal["execute", "continue"]:
    \"\"\"Check if we should batch delete.\"\"\"
    pending = len(state["pending_deletes"])
    batch_size = state["delete_batch_size"]
    producer_active = state["producer_active"]

    # Delete if batch full OR producer done
    if pending >= batch_size or (not producer_active and pending > 0):
        return "execute"
    else:
        return "continue"

def execute_delete(state: StreamingState) -> StreamingState:
    \"\"\"Execute batch IMAP delete.\"\"\"
    to_delete = state["pending_deletes"]

    imap_client.batch_delete(to_delete)

    return {
        **state,
        "pending_deletes": [],  # Clear batch
        "total_deleted": state["total_deleted"] + len(to_delete)
    }
```

### Deadlock Prevention

**Potential Deadlocks:**

1. **Producer waiting + All workers blocked on full delete queue**
   - Solution: Delete queue is unbounded (just a list with reducer)
   - Workers never block, only accumulate

2. **Circular dependency: fetch → process → batch_delete → fetch**
   - Solution: Batch delete is conditional, not blocking
   - Producer can continue fetching while delete pending

3. **Queue full + Workers idle**
   - Solution: Workers process from `email_headers` dict, not queue
   - Queue size is just a counter for backpressure

**Key Invariants:**

- Workers NEVER wait for delete to complete
- Producer only waits when `in_flight_uids` exceeds threshold
- Delete executes independently when batch ready
- No circular waits in graph edges

### Performance Characteristics

**Memory Usage:**
- Bounded by `max_queue_size` (e.g., 1000 emails × 10KB = 10MB)
- Delete batches are lists of UIDs (minimal memory)

**Throughput:**
- Producer fetches in batches of 100 (1 IMAP round-trip)
- N parallel workers process emails
- Delete every M emails (configurable)

**Latency:**
- First email processed after initial fetch (100ms)
- No need to wait for all 150K emails to fetch
- Progressive deletion reduces IMAP server load

## Implementation Plan

1. **Phase 1**: Modify `langgraph_state.py` to add streaming fields
2. **Phase 2**: Rewrite `langgraph_dag.py` with streaming nodes
3. **Phase 3**: Update `imap_client.py` to support windowed fetching
4. **Phase 4**: Add backpressure monitoring and logging
5. **Phase 5**: Test with large mailboxes (10K+ emails)

## Configuration

```python
class StreamingConfig:
    max_queue_size: int = 1000  # Backpressure threshold
    fetch_batch_size: int = 100  # Headers per IMAP fetch
    delete_batch_size: int = 50  # UIDs per delete operation
    parallel_workers: int = 25  # Concurrent email processors
    backpressure_low_watermark: float = 0.3  # Resume at 30% full
```

## Migration from Current Design

Current DAG must be **replaced**, not modified:
- Old: Single batch_fetch → fan-out → aggregate → delete
- New: Loop of (fetch → fan-out → delete coordinator)

Backward compatibility: Not possible - fundamentally different execution model
