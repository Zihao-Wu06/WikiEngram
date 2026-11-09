"""HTTP API (FastAPI).

Long operations (adapt, consolidate, sync with repair) run as background jobs: ``POST`` returns a job id,
``GET /jobs/{id}`` reports progress and the result. One lock serializes everything that touches the model, so
training and inference never run on it at the same time.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..runtime import NeedsAdaptation, WikiEngram
from ..wiki import ClaimStatus


class IngestRequest(BaseModel):
    text: str
    title: str = "note"
    kind: str = Field("document", pattern="^(document|conversation|note)$")


class ConversationRequest(BaseModel):
    messages: list[dict]


class ClaimCreate(BaseModel):
    subject: str
    predicate: str
    object: str
    statement: str | None = None
    importance: float = 0.5
    pinned: bool = False
    confidence: float = 1.0


class ClaimPatch(BaseModel):
    object: str | None = None
    statement: str | None = None
    reason: str = "user correction"


class AskRequest(BaseModel):
    question: str
    mode: str | None = Field(None, pattern="^(none|wiki|engram|hybrid)$")


class ConsolidateRequest(BaseModel):
    claim_ids: list[str] | None = None
    verify: bool = True


class SyncRequest(BaseModel):
    repair: bool = True


class AdaptRequest(BaseModel):
    steps: int | None = None
    corpus: str | None = None
    force: bool = False


def _claim(c) -> dict:
    return c.to_dict()


def create_app(workspace: str | Path) -> FastAPI:
    ws = WikiEngram.open(workspace)
    lock = threading.RLock()
    pool = ThreadPoolExecutor(max_workers=1)
    jobs: dict[str, dict] = {}
    app = FastAPI(title="WikiEngram", version="0.1.0",
                  description="Train Memory, Not Your LLM: LLMWiki explicit memory + Engram neural memory.")

    def submit(kind: str, fn) -> dict:
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"id": job_id, "kind": kind, "status": "queued", "progress": [], "result": None,
                        "error": None, "created_at": time.time()}

        def run():
            jobs[job_id]["status"] = "running"
            try:
                with lock:
                    jobs[job_id]["result"] = fn(lambda rec: jobs[job_id]["progress"].append(rec))
                jobs[job_id]["status"] = "done"
            except Exception as e:  # report, never crash the server
                jobs[job_id].update(status="failed", error=f"{type(e).__name__}: {e}",
                                    traceback=traceback.format_exc(limit=5))

        pool.submit(run)
        return jobs[job_id]

    @app.get("/status")
    def status():
        return ws.status()

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        with lock:
            return ws.wiki.ingest_text(req.text, title=req.title, kind=req.kind).__dict__

    @app.post("/ingest/conversation")
    def ingest_conversation(req: ConversationRequest):
        with lock:
            return ws.wiki.ingest_conversation(req.messages).__dict__

    @app.get("/claims")
    def claims(status: str = "active"):
        st = None if status == "all" else ClaimStatus(status)
        return [_claim(c) for c in ws.wiki.claims(status=st)]

    @app.post("/claims")
    def add_claim(req: ClaimCreate):
        with lock:
            return _claim(ws.wiki.add_claim(req.subject, req.predicate, req.object, req.statement,
                                            confidence=req.confidence, importance=req.importance,
                                            pinned=req.pinned))

    @app.get("/claims/{claim_id}")
    def get_claim(claim_id: str):
        c = ws.wiki.get(claim_id)
        if c is None:
            raise HTTPException(404, "no such claim")
        return _claim(c)

    @app.get("/claims/{claim_id}/history")
    def history(claim_id: str):
        return [_claim(c) for c in ws.wiki.history(claim_id)]

    @app.patch("/claims/{claim_id}")
    def correct(claim_id: str, req: ClaimPatch):
        with lock:
            return _claim(ws.wiki.correct(claim_id, object=req.object, statement=req.statement, reason=req.reason))

    @app.delete("/claims/{claim_id}")
    def delete(claim_id: str, purge: bool = False):
        with lock:
            if purge:
                ws.wiki.purge(claim_id)
                return {"purged": claim_id}
            return _claim(ws.wiki.retract(claim_id))

    @app.get("/search")
    def search(q: str, k: int = 5):
        return [{"score": s, "claim": _claim(c)} for c, s in ws.wiki.search(q, k=k)]

    @app.get("/lint")
    def lint():
        return ws.wiki.lint().__dict__

    @app.post("/ask")
    def ask(req: AskRequest):
        with lock:
            try:
                return ws.ask(req.question, req.mode).to_dict()
            except NeedsAdaptation as e:
                raise HTTPException(409, str(e)) from e

    @app.get("/plan")
    def plan():
        return ws.plan().to_dict()

    @app.post("/consolidate")
    def consolidate(req: ConsolidateRequest):
        return submit("consolidate", lambda cb: ws.consolidate(req.claim_ids, verify=req.verify, progress=cb))

    @app.post("/sync")
    def sync(req: SyncRequest):
        return submit("sync", lambda cb: ws.sync(repair=req.repair, progress=cb))

    @app.post("/adapt")
    def adapt(req: AdaptRequest):
        return submit("adapt", lambda cb: ws.adapt(steps=req.steps, corpus=req.corpus, force=req.force, progress=cb))

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return jobs[job_id]

    @app.get("/versions")
    def versions():
        return ws.versions()

    return app
