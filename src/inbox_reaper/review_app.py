"""Review-then-delete in batches, driven from a local web page.

``inbox-reaper serve`` runs the loop that was otherwise a series of commands:

1. **Start a batch**: take the next ``size`` INBOX UIDs after the last batch
   and dry-run them (``process --dry-run --only-uids --report``). Nothing in
   the mailbox changes.
2. **Review** the proposed deletes on the page (sankey, risk tiers) and mark
   each one Delete or Don't delete. Anything undecided is kept.
3. **Delete** exactly the marked UIDs: they are checked against the batch's
   proposed deletes, the folder's UIDVALIDITY is checked against the one the
   dry run saw (so a UID still names the same email), the pipeline re-checks
   every email before it goes, and afterwards the INBOX is searched to verify
   that the deleted UIDs are gone.

Batch files (reports, delete lists) hold email metadata, so they live in a
per-account directory under ``~/.inbox-reaper`` rather than in a repository.

The HTTP server binds to 127.0.0.1 only. Every API call needs the per-run
token embedded in the page and a 127.0.0.1/localhost Host header, so other
websites (cross-site requests, DNS rebinding) cannot read mail data or
trigger deletes.
"""

import asyncio
import json
import logging
import secrets
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .state import Config
from .timeutil import utcnow

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 2000
MAX_BATCH_SIZE = 10000
STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Persistent batch state
# ---------------------------------------------------------------------------


class BatchStatus(StrEnum):
    DRY_RUN = "dry_run"  # dry run in progress
    READY = "ready"  # report ready for review
    DELETING = "deleting"  # real run in progress
    DONE = "done"  # deleted what was marked (possibly nothing)
    DISCARDED = "discarded"  # dropped before deleting; its UIDs are offered again
    FAILED = "failed"  # dry run failed; nothing deleted


class Batch(BaseModel):
    id: int
    first_uid: int
    last_uid: int
    size: int
    uidvalidity: str | None = None
    status: BatchStatus = BatchStatus.DRY_RUN
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    proposed: int = 0  # proposed deletes in the dry run
    marked: int = 0  # UIDs you marked Delete
    deleted: int = 0  # emails the real run deleted
    remaining: int | None = None  # marked UIDs still in the INBOX afterwards
    inbox_after: int | None = None
    error: str | None = None


class ReviewState(BaseModel):
    version: int = STATE_VERSION
    email: str = ""
    batches: list[Batch] = Field(default_factory=list)

    @property
    def cursor(self) -> int:
        """Highest UID already covered by a batch that was not discarded."""
        live = [
            b.last_uid
            for b in self.batches
            if b.status not in {BatchStatus.DISCARDED, BatchStatus.FAILED}
        ]
        return max(live, default=0)

    @property
    def open_batch(self) -> Batch | None:
        """The batch still awaiting a decision (at most one at a time)."""
        return next(
            (
                b
                for b in self.batches
                if b.status
                in {BatchStatus.DRY_RUN, BatchStatus.READY, BatchStatus.DELETING}
            ),
            None,
        )

    def batch(self, batch_id: int) -> Batch:
        for b in self.batches:
            if b.id == batch_id:
                return b
        raise KeyError(batch_id)


class ReviewStore:
    """Batch state and files for one account, in ``root``."""

    def __init__(self, root: Path, email: str):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = root / "state.json"
        self.lock = threading.Lock()
        if self.path.exists():
            self.state = ReviewState.model_validate_json(self.path.read_text())
        else:
            self.state = ReviewState(email=email)
        if self.state.email and email and self.state.email != email:
            raise ValueError(f"{root} belongs to {self.state.email}, not {email}")
        self._recover()

    def _recover(self) -> None:
        """A server stopped mid-job: an unfinished dry run is discarded; an
        unfinished delete goes back to review (re-running it is safe: deleted
        UIDs are simply gone)."""
        for b in self.state.batches:
            if b.status == BatchStatus.DRY_RUN:
                b.status, b.error = BatchStatus.DISCARDED, "interrupted"
            elif b.status == BatchStatus.DELETING:
                b.status, b.error = BatchStatus.READY, "delete was interrupted"
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(self.state.model_dump_json(indent=1))
        tmp.replace(self.path)

    def dir(self, batch_id: int) -> Path:
        path = self.root / f"batch-{batch_id:04d}"
        path.mkdir(exist_ok=True)
        return path

    def report_rows(self, batch_id: int) -> list[dict[str, Any]]:
        path = self.dir(batch_id) / "report.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def totals(self) -> dict[str, int]:
        done = [b for b in self.state.batches if b.status == BatchStatus.DONE]
        return {
            "batches": len(done),
            "reviewed": sum(b.size for b in done),
            "deleted": sum(b.deleted for b in done),
        }


# ---------------------------------------------------------------------------
# Mailbox operations (real ones use IMAP; tests inject fakes)
# ---------------------------------------------------------------------------


@dataclass
class Survey:
    uidvalidity: str | None
    uids: list[int]  # the next batch
    inbox_total: int
    remaining_after: int  # unreviewed UIDs after this batch


class MailOps(Protocol):
    async def survey(self, after_uid: int, size: int) -> Survey: ...
    async def inbox(self) -> tuple[str | None, set[str]]: ...
    async def dry_run(self, uids_path: Path, report_path: Path) -> None: ...
    async def delete(self, uids_path: Path, report_path: Path) -> None: ...


class IMAPOps:
    """MailOps over the real pipeline; the AI stays off unless ``use_ai``."""

    def __init__(self, config: Config):
        self.config = config

    async def _uids(self) -> tuple[str | None, list[int]]:
        from .langgraph_streaming import IMAPSession  # noqa: PLC0415

        async with IMAPSession(self.config.email) as session:

            async def survey(client: Any) -> tuple[str | None, list[str]]:
                validity = await client.examine_mailbox("INBOX")
                return validity, await client.search_uids("ALL")

            validity, uids = await session.run(survey)
        return validity, sorted(int(u) for u in uids)

    async def survey(self, after_uid: int, size: int) -> Survey:
        validity, uids = await self._uids()
        later = [u for u in uids if u > after_uid]
        return Survey(validity, later[:size], len(uids), max(0, len(later) - size))

    async def inbox(self) -> tuple[str | None, set[str]]:
        validity, uids = await self._uids()
        return validity, {str(u) for u in uids}

    async def _run(self, uids_path: Path, report_path: Path, *, dry_run: bool) -> None:
        from .langgraph_streaming import run_streaming_pipeline  # noqa: PLC0415

        await run_streaming_pipeline(
            self.config.model_copy(
                update={
                    "dry_run": dry_run,
                    "only_uids_path": str(uids_path),
                    "report_path": str(report_path),
                    "max_emails": None,
                }
            )
        )

    async def dry_run(self, uids_path: Path, report_path: Path) -> None:
        await self._run(uids_path, report_path, dry_run=True)

    async def delete(self, uids_path: Path, report_path: Path) -> None:
        await self._run(uids_path, report_path, dry_run=False)


# ---------------------------------------------------------------------------
# Jobs: one background job at a time
# ---------------------------------------------------------------------------


class BusyError(Exception):
    """Another job is running, or a batch is still awaiting a decision."""


class RejectedError(Exception):
    """The request would be unsafe or makes no sense in the current state."""


@dataclass
class Job:
    kind: str  # "dry_run" | "delete"
    batch_id: int
    running: bool = True
    message: str = ""
    error: str | None = None
    started_at: datetime = field(default_factory=utcnow)


class ReviewService:
    """Batch workflow on top of a store and mailbox operations."""

    def __init__(self, store: ReviewStore, ops: MailOps):
        self.store = store
        self.ops = ops
        self.job: Job | None = None
        self._thread: threading.Thread | None = None

    # -- helpers -----------------------------------------------------------

    def _start(self, job: Job, work: Callable[[], Any]) -> None:
        def run() -> None:
            try:
                asyncio.run(work())
                job.message = job.message or "done"
            except Exception as e:  # reported to the page, never swallowed
                logger.exception("%s job for batch %s failed", job.kind, job.batch_id)
                job.error = str(e) or type(e).__name__
            finally:
                job.running = False

        self.job = job
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def wait(self, timeout: float = 30) -> None:
        """Block until the current job finishes (tests and shutdown)."""
        if self._thread:
            self._thread.join(timeout)

    def _check_idle(self) -> None:
        if self.job and self.job.running:
            raise BusyError(f"a {self.job.kind} job is still running")

    # -- actions -----------------------------------------------------------

    def start_batch(self, size: int) -> Batch:
        if not 1 <= size <= MAX_BATCH_SIZE:
            raise RejectedError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")
        with self.store.lock:
            self._check_idle()
            if open_batch := self.store.state.open_batch:
                raise BusyError(f"batch {open_batch.id} is still awaiting a decision")
            state = self.store.state
            batch = Batch(
                id=max((b.id for b in state.batches), default=0) + 1,
                first_uid=state.cursor + 1,
                last_uid=state.cursor,
                size=0,
            )
            state.batches.append(batch)
            self.store.save()
            job = Job("dry_run", batch.id, message="finding the next emails")

        async def work() -> None:
            try:
                await self._dry_run(batch, size, job)
            except Exception as e:
                with self.store.lock:
                    batch.status, batch.error = BatchStatus.FAILED, str(e)
                    batch.finished_at = utcnow()
                    self.store.save()
                raise

        self._start(job, work)
        return batch

    async def _dry_run(self, batch: Batch, size: int, job: Job) -> None:
        state = self.store.state
        known = {b.uidvalidity for b in state.batches if b.uidvalidity} - {None}
        survey = await self.ops.survey(state.cursor, size)
        if known and survey.uidvalidity not in known:
            raise RejectedError(
                "the INBOX was renumbered (UIDVALIDITY changed) since earlier "
                "batches; start over with a new state directory"
            )
        if not survey.uids:
            raise RejectedError("no unreviewed emails left")
        directory = self.store.dir(batch.id)
        uids_path = directory / "uids.txt"
        uids_path.write_text("".join(f"{u}\n" for u in survey.uids))
        with self.store.lock:
            batch.first_uid, batch.last_uid = survey.uids[0], survey.uids[-1]
            batch.size, batch.uidvalidity = len(survey.uids), survey.uidvalidity
            self.store.save()
        job.message = f"dry run of {len(survey.uids)} emails"
        await self.ops.dry_run(uids_path, directory / "report.jsonl")
        rows = self.store.report_rows(batch.id)
        with self.store.lock:
            batch.proposed = sum(r.get("operation") == "delete" for r in rows)
            batch.status = BatchStatus.READY
            self.store.save()
        job.message = (
            f"{batch.proposed} proposed deletes in {batch.size} emails; "
            f"{survey.remaining_after} unreviewed after this batch"
        )

    def delete(self, batch_id: int, uids: list[str]) -> Batch:
        with self.store.lock:
            self._check_idle()
            batch = self._batch(batch_id)
            if batch.status != BatchStatus.READY:
                raise RejectedError(f"batch {batch_id} is {batch.status}, not ready")
            proposed = {
                str(r["uid"])
                for r in self.store.report_rows(batch_id)
                if r.get("operation") == "delete"
            }
            marked = sorted({str(u) for u in uids}, key=int)
            if outside := [u for u in marked if u not in proposed]:
                raise RejectedError(
                    f"{len(outside)} UIDs are not proposed deletes of batch "
                    f"{batch_id} (e.g. {outside[0]})"
                )
            path = self.store.dir(batch_id) / "delete.txt"
            path.write_text("".join(f"{u}\n" for u in marked))
            batch.marked = len(marked)
            if not marked:  # nothing to delete: the batch is simply finished
                batch.status, batch.finished_at = BatchStatus.DONE, utcnow()
                self.store.save()
                self.job = Job("delete", batch_id, running=False, message="kept all")
                return batch
            batch.status = BatchStatus.DELETING
            self.store.save()
            job = Job("delete", batch_id, message=f"deleting {len(marked)} emails")

        async def work() -> None:
            try:
                await self._delete(batch, set(marked), path, job)
            except Exception as e:
                with self.store.lock:
                    batch.status, batch.error = BatchStatus.READY, str(e)
                    self.store.save()
                raise

        self._start(job, work)
        return batch

    async def _delete(
        self, batch: Batch, marked: set[str], path: Path, job: Job
    ) -> None:
        validity, _ = await self.ops.inbox()
        if batch.uidvalidity and validity != batch.uidvalidity:
            raise RejectedError(
                "the INBOX was renumbered since this batch's dry run "
                "(UIDVALIDITY changed): nothing was deleted"
            )
        report = self.store.dir(batch.id) / "deleted.jsonl"
        await self.ops.delete(path, report)
        deleted = {
            str(json.loads(line)["uid"])
            for line in (report.read_text().splitlines() if report.exists() else [])
            if line and json.loads(line).get("operation") == "delete"
        }
        if not deleted <= marked:  # cannot happen with --only-uids; never hide it
            raise RejectedError(
                f"deleted UIDs outside the marked set: {deleted - marked}"
            )
        _, inbox = await self.ops.inbox()
        with self.store.lock:
            batch.deleted = len(deleted)
            batch.remaining = len(marked & inbox)
            batch.inbox_after = len(inbox)
            batch.status, batch.finished_at = BatchStatus.DONE, utcnow()
            batch.error = None
            self.store.save()
        job.message = (
            f"deleted {batch.deleted} of {len(marked)} marked; "
            f"{batch.remaining} still in the INBOX; INBOX now {len(inbox)}"
        )

    def discard(self, batch_id: int) -> Batch:
        with self.store.lock:
            self._check_idle()
            batch = self._batch(batch_id)
            if batch.status != BatchStatus.READY:
                raise RejectedError(f"batch {batch_id} is {batch.status}, not ready")
            batch.status, batch.finished_at = BatchStatus.DISCARDED, utcnow()
            self.store.save()
            return batch

    def _batch(self, batch_id: int) -> Batch:
        try:
            return self.store.state.batch(batch_id)
        except KeyError:
            raise RejectedError(f"no batch {batch_id}") from None

    def snapshot(self) -> dict[str, Any]:
        with self.store.lock:
            state = self.store.state
            return {
                "email": state.email,
                "batches": [b.model_dump(mode="json") for b in state.batches],
                "open_batch": state.open_batch.id if state.open_batch else None,
                "cursor": state.cursor,
                "totals": self.store.totals(),
                "job": (
                    {
                        "kind": self.job.kind,
                        "batch_id": self.job.batch_id,
                        "running": self.job.running,
                        "message": self.job.message,
                        "error": self.job.error,
                    }
                    if self.job
                    else None
                ),
            }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

PAGE = "review.html"
MAX_BODY = 2_000_000


def page_html(token: str, settings: dict[str, Any]) -> bytes:
    template = resources.files("inbox_reaper.web").joinpath(PAGE).read_text()
    boot = json.dumps({"token": token, **settings}).replace("<", "\\u003c")
    return template.replace("/*__BOOT__*/null", boot).encode()


def _delete_request(body: dict[str, Any]) -> list[str]:
    """UIDs from a delete request, which must carry explicit confirmation."""
    uids = body.get("uids")
    if not isinstance(uids, list):
        raise RejectedError("expected a list of uids")
    if body.get("confirm") != "delete":
        raise RejectedError("deleting needs explicit confirmation")
    return [str(u) for u in uids]


class ReviewHandler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("%s %s", self.address_string(), format % args)

    # -- plumbing ----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _host_ok(self) -> bool:
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _token_ok(self) -> bool:
        given = self.headers.get("X-Reaper-Token", "")
        return secrets.compare_digest(given, self.server.token)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise RejectedError("request too large")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise RejectedError("expected a JSON object")
        return data

    def _guard(self) -> bool:
        if not self._host_ok():
            self._error(HTTPStatus.FORBIDDEN, "bad host")
            return False
        if self.path.startswith("/api/") and not self._token_ok():
            self._error(HTTPStatus.FORBIDDEN, "missing or wrong token")
            return False
        return True

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:
        if not self._guard():
            return
        service = self.server.service
        parts = self.path.split("?")[0].strip("/").split("/")
        if parts == [""]:
            self._send(
                HTTPStatus.OK,
                page_html(self.server.token, self.server.settings),
                "text/html; charset=utf-8",
            )
        elif parts == ["api", "state"]:
            self._json(HTTPStatus.OK, service.snapshot())
        elif parts[:2] == ["api", "batches"] and parts[3:] == ["report"]:
            try:
                batch_id = int(parts[2])
                service._batch(batch_id)
            except (ValueError, RejectedError):
                self._error(HTTPStatus.NOT_FOUND, "no such batch")
                return
            self._json(HTTPStatus.OK, service.store.report_rows(batch_id))
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if not self._guard():
            return
        service = self.server.service
        parts = self.path.split("?")[0].strip("/").split("/")
        try:
            body = self._body()
            if parts == ["api", "batches"]:
                size = int(body.get("size", self.server.settings["batch_size"]))
                result = service.start_batch(size)
            elif len(parts) == 4 and parts[:2] == ["api", "batches"]:
                batch_id = int(parts[2])
                if parts[3] == "delete":
                    result = service.delete(batch_id, _delete_request(body))
                elif parts[3] == "discard":
                    result = service.discard(batch_id)
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                    return
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
        except BusyError as e:
            self._error(HTTPStatus.CONFLICT, str(e))
            return
        except (RejectedError, ValueError, json.JSONDecodeError) as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))
            return
        self._json(HTTPStatus.OK, result.model_dump(mode="json"))


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, service: ReviewService, settings: dict[str, Any], port: int = 0):
        super().__init__(("127.0.0.1", port), ReviewHandler)
        self.service = service
        self.settings = settings
        self.token = secrets.token_urlsafe(32)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


def default_state_dir(email: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "@._-" else "_" for c in email)
    return Path.home() / ".inbox-reaper" / safe / "review"


def serve(
    config: Config,
    *,
    state_dir: Path | None = None,
    port: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    open_browser: bool = True,
) -> None:
    """Run the review app until interrupted."""
    store = ReviewStore(state_dir or default_state_dir(config.email), config.email)
    service = ReviewService(store, IMAPOps(config))
    settings = {
        "email": config.email,
        "batch_size": batch_size,
        "combiner": bool(config.combiner_path),
        "ai": config.use_ai,
    }
    server = ReviewServer(service, settings, port)
    print(f"Review app for {config.email}: {server.url}")
    print(f"Batch data: {store.root}")
    print("Press Ctrl-C to stop.")
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping (waiting for a running job to finish)...")
        service.wait(timeout=600)
    finally:
        server.server_close()
