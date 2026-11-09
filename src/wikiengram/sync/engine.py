"""Memory sync: propagate wiki corrections and deletions into the Engram delta, then verify.

For every wiki change event since the last sync that touches a consolidated claim (updated, retracted,
restored, purged):

1. **Forget** the old memory: rows owned only by that claim are removed from the delta (they read as the frozen
   base table again, bit for bit); rows shared with other claims are removed too (``repair=True``) or masked
   (``repair=False``, instant, no training).
2. **Repair**: the surviving claims that shared those rows are retrained, restricted to exactly those rows.
3. **Verify**: Engram alone must no longer produce the old answer, and the repaired claims must still be
   recalled. The new version of an edited claim is answered from the wiki immediately and consolidated later
   by the normal policy.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch

from ..consolidation.samples import Probe
from ..consolidation.trainer import DeltaTrainConfig, train_delta
from ..consolidation.verify import VerificationReport, run_probes
from ..prompts import qa_prompt
from ..wiki import EngramState

FORGET_KINDS = {"updated", "retracted", "restored", "purged"}


def run_sync(ws, repair: bool = True, verify: bool = True, progress=None) -> dict:
    t0 = time.perf_counter()
    events = ws.pending_events()
    has_pending_repair = bool(ws.state.pending_repair)
    if not events and not (repair and has_pending_repair):
        return {"events": 0, "forgotten": [], "repaired": [], "seconds": 0.0}

    touched: list[str] = []
    for e in events:
        if e.kind in FORGET_KINDS and e.claim_id not in touched:
            touched.append(e.claim_id)

    forgotten: list[str] = []
    forget_probes: list[Probe] = []
    shared_rows: dict[int, set[int]] = defaultdict(set)
    affected: set[str] = set()
    memory_changed = False
    for cid in touched:
        if cid not in ws.ledger:
            continue  # never consolidated (or already forgotten): the wiki alone is enough
        info = ws.state.claims.get(cid, {})
        current = ws.wiki.get(cid)
        old_obj = info.get("object")
        still_true = (current is not None and getattr(current.status, "value", current.status) == "active"
                      and current.object == old_obj)
        if old_obj and not still_true:
            for q in info.get("questions", [])[:2]:
                forget_probes.append(Probe(cid, qa_prompt(q), old_obj, "forget"))
        shared, aff = ws._forget_rows(cid, repair=repair)
        for lid, rows in shared.items():
            shared_rows[int(lid)].update(int(r) for r in rows.tolist())
        affected |= aff
        ws.state.claims.pop(cid, None)
        if current is not None:
            ws.wiki.set_engram_state(cid, EngramState.NONE, None)
        forgotten.append(cid)
        memory_changed = True
    affected -= set(forgotten)

    repaired: list[str] = []
    repair_report = None
    lazy_check = None
    if repair:
        lm = ws.lm
        for lid_s, rows in ws.state.pending_repair.items():  # rows masked by an earlier instant sync
            lid = int(lid_s)
            arr = np.asarray(sorted(rows), dtype=np.int64)
            lm.delta.layers[lid].remove(torch.from_numpy(arr))
            shared_rows[lid].update(rows)
            affected |= ws.ledger.owners(lid, arr)
        ws.state.pending_repair.clear()
        affected -= set(forgotten)
        restrict = {lid: np.asarray(sorted(r), dtype=np.int64) for lid, r in shared_rows.items() if r}
        needs = sorted(affected)
        if restrict and needs and ws.config.consolidation.lazy_repair:
            # most affected claims lost only a row or two to a hash collision and still recall fine:
            # retrain only those whose core probes (trained questions, statement) now fail
            lazy_check = run_probes(lm, ws.probes_for(needs))
            core = VerificationReport([r for r in lazy_check.results if r.kind != "heldout_question"]).per_claim()
            needs = [cid for cid in needs if not core.get(cid, False)]
        samples = ws.samples_for(needs) if restrict else []
        if samples:
            # the lost shared rows alone may not be enough to restore recall: let each failing claim also
            # retrain the rows it owns exclusively (that cannot disturb any other claim)
            for cid in needs:
                exclusive, _, _ = ws.ledger.split_for_removal(cid)
                for lid, rows in exclusive.items():
                    restrict[lid] = np.union1d(restrict.get(lid, np.empty(0, np.int64)), rows)
        if samples:
            c = ws.config.consolidation
            blocked, _ = ws.generic_rows(samples)
            # ledger=: every row a repair writes is recorded for its claim, so deleting it later stays exact
            repair_report = train_delta(
                lm, lm.delta, samples,
                DeltaTrainConfig(epochs=c.epochs, batch_size=c.batch_size, lr=c.lr),
                ledger=ws.ledger, blocked_rows=blocked, restrict_rows=restrict, progress=progress,
            ).to_dict()
            repaired = sorted({s.claim_id for s in samples})
            memory_changed = True

    verification = {}
    survivors = sorted(c for c in affected if c in ws.state.claims)
    if verify and (forget_probes or survivors) and (repair or not shared_rows):
        lm = ws.lm
        forget = run_probes(lm, forget_probes, expect_absent=True) if forget_probes else VerificationReport()
        retain = run_probes(lm, ws.probes_for(repaired)) if repaired else VerificationReport()
        verification = {"forget": forget.summary(), "retain_repaired": retain.summary()}
        # every claim that lost rows is re-verified on its full probe set (repaired ones after the repair)
        per_claim = dict(lazy_check.per_claim()) if lazy_check else {}
        others = [c for c in survivors if c not in per_claim and c not in repaired]
        if others:
            per_claim.update(run_probes(lm, ws.probes_for(others)).per_claim())
        per_claim.update(retain.per_claim())
        for cid in survivors:
            ws.state.claims[cid]["verified"] = bool(per_claim.get(cid, False))
    else:
        # not verified now (instant sync or verify=False): these claims must not be trusted by hybrid routing
        for cid in survivors:
            ws.state.claims[cid]["verified"] = False

    if events:
        ws.state.last_event_seq = max(e.seq for e in events)
    purged = any(e.kind == "purged" for e in events)
    if purged:
        ws._drop_history()  # old snapshots still hold the purged claim's rows and text
    if memory_changed or purged:
        ws._save_memory("sync")
    else:
        ws.state.save(ws.state_path)
    report = {"events": len(events), "forgotten": forgotten, "repaired": repaired,
              "affected": sorted(affected), "lazy_check": lazy_check.summary() if lazy_check else None,
              "shared_rows": {k: len(v) for k, v in shared_rows.items()}, "repair_training": repair_report,
              "verification": verification, "masked_pending_repair": not repair and bool(shared_rows),
              "seconds": time.perf_counter() - t0}
    ws._write_run("sync", report)
    return report
