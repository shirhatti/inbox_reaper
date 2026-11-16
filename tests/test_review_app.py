"""Tests for the batch review web app (fake mailbox, no IMAP)."""

import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inbox_reaper.review_app import (
    BatchStatus,
    BusyError,
    RejectedError,
    ReviewServer,
    ReviewService,
    ReviewStore,
    Survey,
)
from inbox_reaper.state import Email


def read_uids(path: Path) -> list[str]:
    return path.read_text().split()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class FakeOps:
    """A mailbox of UIDs 1..n; the 'pipeline' proposes deleting even UIDs."""

    def __init__(self, n: int = 10, validity: str = "7"):
        self.uids = set(range(1, n + 1))
        self.validity = validity
        self.deleted_calls: list[list[str]] = []

    async def survey(self, after_uid: int, size: int) -> Survey:
        later = sorted(u for u in self.uids if u > after_uid)
        return Survey(
            self.validity, later[:size], len(self.uids), max(0, len(later) - size)
        )

    async def inbox(self) -> tuple[str | None, set[str]]:
        return self.validity, {str(u) for u in self.uids}

    async def dry_run(self, uids_path: Path, report_path: Path) -> None:
        rows = [
            {
                "uid": u,
                "operation": "delete" if int(u) % 2 == 0 else "keep",
                "reason": "combiner" if int(u) % 2 == 0 else "undecided",
                "date": "2015-01-01T00:00:00+00:00",
                "sender": f"news{u}@shop.example",
                "subject": f"Deal {u}",
            }
            for u in read_uids(uids_path)
        ]
        write_rows(report_path, rows)

    async def delete(self, uids_path: Path, report_path: Path) -> None:
        uids = read_uids(uids_path)
        self.deleted_calls.append(uids)
        self.uids -= {int(u) for u in uids}
        write_rows(report_path, [{"uid": u, "operation": "delete"} for u in uids])


def make_service(tmp_path, ops=None) -> ReviewService:
    return ReviewService(
        ReviewStore(tmp_path / "state", "me@example.com"), ops or FakeOps()
    )


def ready_batch(service: ReviewService, size: int = 4):
    batch = service.start_batch(size)
    service.wait()
    assert service.job.error is None, service.job.error
    return batch


class TestBatches:
    def test_dry_run_makes_a_ready_batch(self, tmp_path):
        service = make_service(tmp_path)
        batch = ready_batch(service)
        assert batch.status == BatchStatus.READY
        assert (batch.first_uid, batch.last_uid, batch.size) == (1, 4, 4)
        assert batch.proposed == 2  # UIDs 2 and 4
        assert service.store.state.cursor == 4
        assert "2 proposed deletes" in service.job.message

    def test_one_open_batch_at_a_time(self, tmp_path):
        service = make_service(tmp_path)
        ready_batch(service)
        with pytest.raises(BusyError, match="awaiting a decision"):
            service.start_batch(4)

    def test_delete_exactly_the_marked_uids_and_verify(self, tmp_path):
        ops = FakeOps()
        service = make_service(tmp_path, ops)
        batch = ready_batch(service)
        service.delete(batch.id, ["4"])  # keep proposed UID 2
        service.wait()
        assert service.job.error is None
        assert ops.deleted_calls == [["4"]]
        assert batch.status == BatchStatus.DONE
        assert (batch.marked, batch.deleted, batch.remaining) == (1, 1, 0)
        assert batch.inbox_after == 9 and 2 in ops.uids
        # The next batch starts after this one
        nxt = ready_batch(service)
        assert (nxt.first_uid, nxt.last_uid) == (5, 8)
        assert service.store.totals() == {"batches": 1, "reviewed": 4, "deleted": 1}

    def test_rejects_uids_that_were_not_proposed(self, tmp_path):
        ops = FakeOps()
        service = make_service(tmp_path, ops)
        batch = ready_batch(service)
        for bad in (["3"], ["2", "9"]):  # a keep, and a UID outside the batch
            with pytest.raises(RejectedError, match="not proposed deletes"):
                service.delete(batch.id, bad)
        assert ops.deleted_calls == [] and batch.status == BatchStatus.READY

    def test_marking_nothing_finishes_without_deleting(self, tmp_path):
        ops = FakeOps()
        service = make_service(tmp_path, ops)
        batch = ready_batch(service)
        service.delete(batch.id, [])
        assert batch.status == BatchStatus.DONE and ops.deleted_calls == []

    def test_renumbered_inbox_blocks_the_delete(self, tmp_path):
        ops = FakeOps()
        service = make_service(tmp_path, ops)
        batch = ready_batch(service)
        ops.validity = "8"  # UIDs may now name different emails
        service.delete(batch.id, ["2"])
        service.wait()
        assert "renumbered" in service.job.error
        assert ops.deleted_calls == []
        assert batch.status == BatchStatus.READY

    def test_discard_offers_the_same_emails_again(self, tmp_path):
        service = make_service(tmp_path)
        batch = ready_batch(service)
        service.discard(batch.id)
        assert batch.status == BatchStatus.DISCARDED
        again = ready_batch(service)
        assert (again.first_uid, again.last_uid) == (1, 4)

    def test_nothing_left_fails_cleanly(self, tmp_path):
        service = make_service(tmp_path, FakeOps(n=0))
        batch = service.start_batch(4)
        service.wait()
        assert "no unreviewed emails" in service.job.error
        assert batch.status == BatchStatus.FAILED
        assert service.store.state.open_batch is None

    def test_state_survives_restart_and_recovers_interrupted_jobs(self, tmp_path):
        service = make_service(tmp_path)
        done = ready_batch(service)
        service.delete(done.id, ["2"])
        service.wait()
        open_one = ready_batch(service)
        # Simulate a server killed mid-delete
        open_one.status = BatchStatus.DELETING
        service.store.save()
        reloaded = ReviewStore(tmp_path / "state", "me@example.com")
        assert reloaded.state.batch(done.id).status == BatchStatus.DONE
        assert reloaded.state.batch(open_one.id).status == BatchStatus.READY
        assert reloaded.state.cursor == 8

    def test_state_dir_belongs_to_one_account(self, tmp_path):
        make_service(tmp_path)
        with pytest.raises(ValueError, match="belongs to"):
            ReviewStore(tmp_path / "state", "someone@else.example")


@pytest.fixture
def server(tmp_path):
    ops = FakeOps()
    service = make_service(tmp_path, ops)
    srv = ReviewServer(
        service,
        {"email": "me@example.com", "batch_size": 4, "combiner": True, "ai": False},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, ops
    srv.shutdown()
    srv.server_close()


def call(srv, path, body=None, *, token=True, host=None):
    request = urllib.request.Request(
        srv.url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    if token:
        request.add_header("X-Reaper-Token", srv.token)
    if host:
        request.add_header("Host", host)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            kind = response.headers.get("Content-Type", "")
            return response.status, json.loads(raw) if "json" in kind else raw.decode()
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestHTTP:
    def test_page_carries_the_token_and_settings(self, server):
        srv, _ = server
        status, html = call(srv, "/", token=False)
        assert status == 200
        assert srv.token in html and '"batch_size": 4' in html
        assert "/*__BOOT__*/null" not in html

    def test_api_needs_the_token_and_a_local_host(self, server):
        srv, _ = server
        assert call(srv, "/api/state", token=False)[0] == 403
        assert call(srv, "/api/state", host="evil.example")[0] == 403
        assert call(srv, "/", token=False, host="evil.example:80")[0] == 403
        assert call(srv, "/api/state")[0] == 200

    def test_full_cycle_over_http(self, server):
        srv, ops = server
        status, batch = call(srv, "/api/batches", {"size": 4})
        assert status == 200
        srv.service.wait()
        state = call(srv, "/api/state")[1]
        assert state["open_batch"] == batch["id"]
        rows = call(srv, f"/api/batches/{batch['id']}/report")[1]
        assert [r["uid"] for r in rows if r["operation"] == "delete"] == ["2", "4"]
        # Deleting needs explicit confirmation
        assert (
            call(srv, f"/api/batches/{batch['id']}/delete", {"uids": ["2"]})[0] == 400
        )
        status, _ = call(
            srv,
            f"/api/batches/{batch['id']}/delete",
            {"uids": ["2"], "confirm": "delete"},
        )
        assert status == 200
        srv.service.wait()
        state = call(srv, "/api/state")[1]
        assert state["batches"][0]["status"] == "done"
        assert state["totals"]["deleted"] == 1 and ops.deleted_calls == [["2"]]

    def test_conflicts_and_bad_requests(self, server):
        srv, _ = server
        call(srv, "/api/batches", {"size": 4})
        srv.service.wait()
        assert call(srv, "/api/batches", {"size": 4})[0] == 409
        assert (
            call(srv, "/api/batches/99/delete", {"uids": [], "confirm": "delete"})[0]
            == 400
        )
        assert call(srv, "/api/batches/1/report?x")[0] == 200
        assert call(srv, "/api/batches/zz/report")[0] == 404
        assert call(srv, "/api/batches", {"size": 0})[0] in (400, 409)


# ---------------------------------------------------------------------------
# Training (sample, label, train) from the app
# ---------------------------------------------------------------------------


def fake_email(uid: int) -> Email:
    promo = uid % 2 == 0
    return Email(
        uid=str(uid),
        subject="Weekly deals" if promo else "Re: dinner",
        sender="news@e.shop.example" if promo else "Alex <alex@gmail.com>",
        body="Save 50% today" if promo else "See you at 7",
        date=datetime(2015, 1, 1, tzinfo=UTC),
        headers={"precedence": "bulk", "list-id": "<x>"} if promo else {},
        has_list_unsubscribe=promo,
    )


class TrainingOps(FakeOps):
    def __init__(self, n: int = 200):
        super().__init__(n)
        self.combiner = None

    async def sample(self, count, seed, exclude):
        pool = [u for u in sorted(self.uids) if str(u) not in exclude]
        return [fake_email(u) for u in pool[:count]]

    def set_combiner(self, path):
        self.combiner = path


def label_all(service):
    for s in service.training_samples():
        action = "delete" if int(s["uid"]) % 2 == 0 else "keep"
        service.save_label({"uid": s["uid"], "action": action, "category": "promo"})


class TestTraining:
    def test_sampling_adds_new_emails_only(self, tmp_path):
        service = make_service(tmp_path, TrainingOps())
        service.start_sampling(30)
        service.wait()
        service.start_sampling(30)
        service.wait()
        uids = [s["uid"] for s in service.training_samples()]
        assert len(uids) == len(set(uids)) == 60
        assert service.training.summary()["samples"] == 60

    def test_label_validation_and_relabelling(self, tmp_path):
        service = make_service(tmp_path, TrainingOps())
        service.start_sampling(5)
        service.wait()
        with pytest.raises(RejectedError, match="no sampled email"):
            service.save_label({"uid": "999", "action": "keep"})
        with pytest.raises(RejectedError, match="unknown category"):
            service.save_label({"uid": "1", "action": "keep", "category": "x"})
        with pytest.raises(RejectedError):
            service.save_label({"uid": "1", "action": "maybe"})
        service.save_label({"uid": "1", "action": "delete"})
        service.save_label({"uid": "1", "action": "keep", "loss": "bad"})
        summary = service.training.summary()
        assert (summary["labelled"], summary["keep"], summary["delete"]) == (1, 1, 0)

    def test_training_needs_enough_of_each_label(self, tmp_path):
        service = make_service(tmp_path, TrainingOps())
        service.start_sampling(12)
        service.wait()
        label_all(service)  # 6 keeps, 6 deletes
        assert not service.training.summary()["can_train"]
        with pytest.raises(RejectedError, match="at least 10"):
            service.start_training()

    def test_trained_combiner_is_saved_and_used_for_next_batches(self, tmp_path):
        ops = TrainingOps()
        service = make_service(tmp_path, ops)
        service.start_sampling(80)
        service.wait()
        label_all(service)
        service.start_training()
        service.wait()
        assert service.job.error is None, service.job.error
        assert "wrong deletes in cross-validation" in service.job.message
        path = service.training.combiner_path
        assert path.exists() and ops.combiner == path
        info = service.snapshot()
        assert info["combiner_active"] is True
        assert info["training"]["combiner"]["trained_on"] == 80
        assert info["training"]["combiner"]["metrics"]["false_deletes"] == 0

    def test_training_over_http(self, tmp_path):
        ops = TrainingOps()
        srv = ReviewServer(
            make_service(tmp_path, ops),
            {"email": "me@example.com", "batch_size": 4, "ai": False, "categories": []},
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            assert call(srv, "/api/training/sample", {"count": 3})[0] == 200
            srv.service.wait()
            status, samples = call(srv, "/api/training/samples")
            assert status == 200 and len(samples) == 3 and samples[0]["label"] is None
            status, label = call(
                srv,
                "/api/training/labels",
                {"uid": samples[0]["uid"], "action": "keep"},
            )
            assert status == 200 and label["labeler"] == "human"
            assert call(srv, "/api/training/train", {})[0] == 400  # too few labels
            assert call(srv, "/api/training/samples", token=False)[0] == 403
        finally:
            srv.shutdown()
            srv.server_close()
