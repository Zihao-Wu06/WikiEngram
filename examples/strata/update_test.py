"""Corrections and deletions in the trained Strata memory (Qwen3.8-Flash-Next), measured like the benchmark.

    python update_test.py --workspace WS --out update_results.json

Run it on a copy of the trained workspace: it changes the memory. Steps:

1. In the wiki, correct ``--edit`` facts whose answer contains a number (the number goes up by one, as when the
   documentation changes: version 0.2.13 becomes 0.2.14) and delete ``--delete`` others.
2. ``sync``: the rows only those facts wrote are removed from the personal delta, shared rows are reset, and
   facts that then fail their probes are retrained on those rows.
3. Measure, with each fact's first trained phrasing: corrected facts answered with the new value (LLMWiki only,
   Engram only, LLMWiki + Engram); the old value still produced; deleted facts still produced; untouched facts
   still answered.
4. Consolidate again (the corrected facts at their new versions) and measure the corrected and untouched facts
   with Engram alone.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from evaluate import correct, run_jobs


def new_value(obj: str) -> str:
    """The answer with its last number one higher (``4 to 6`` -> ``4 to 7``, ``0.2.13`` -> ``0.2.14``)."""
    m = list(re.finditer(r"\d+", obj))[-1]
    return obj[: m.start()] + str(int(m.group()) + 1) + obj[m.end():]


def measure(ws, items: dict[str, list[tuple[str, str, list[tuple[str, str]]]]], batch: int) -> dict:
    """``items[mode]``: (claim id, question, [(score name, expected answer)]) -> {score name: accuracy}."""
    from wikiengram.fusion.engine import prepare

    jobs = []
    for mode, rows in items.items():
        for cid, q, scores in rows:
            p = prepare(ws, q, mode)
            jobs.append({"mode": mode, "claim": cid, "question": q, "answer": scores[0][1], "scores": scores,
                         "prompt": p.prompt, "ctx": p.context_tokens, "use_engram": p.use_engram,
                         "use_delta": p.use_delta})
    run_jobs(ws, jobs, batch, 32)
    out, detail = {}, []
    for j in jobs:
        for name, expected in j["scores"]:
            out.setdefault(f"{name}/{j['mode']}", []).append(correct(j["prediction"], expected))
        detail.append({k: j[k] for k in ("mode", "claim", "question", "prediction", "use_delta")} |
                      {"scores": {n: correct(j["prediction"], e) for n, e in j["scores"]}})
    return {k: round(sum(v) / len(v), 4) for k, v in out.items()} | {"_detail": detail}


def run(ws, n_edit: int = 15, n_delete: int = 10, batch: int = 64, seed: int = 0) -> tuple[dict, dict]:
    """The whole test on an open workspace: ``(results, per-question detail)``."""
    claims = {c.id: c for c in ws.wiki.claims()}
    held = sorted(cid for cid in ws.state.claims if cid in claims and cid in ws.ledger)
    question = {cid: ws.state.claims[cid]["questions"][0] for cid in held}
    rng = random.Random(seed)
    numeric = [cid for cid in held if re.search(r"\d", claims[cid].object) and claims[cid].object in
               claims[cid].statement]
    edit = sorted(rng.sample(numeric, min(n_edit, len(numeric))))
    rest = [cid for cid in held if cid not in edit]
    delete = sorted(rng.sample(rest, min(n_delete, len(rest))))
    keep = [cid for cid in held if cid not in edit and cid not in delete]

    old = {cid: claims[cid].object for cid in edit + delete}
    new = {cid: new_value(old[cid]) for cid in edit}
    for cid in edit:
        c = claims[cid]
        ws.wiki.correct(cid, object=new[cid], statement=c.statement.replace(c.object, new[cid]),
                        reason="update test: the documentation changed")
    for cid in delete:
        ws.wiki.retract(cid, reason="update test: removed from the documentation")

    t = time.perf_counter()
    sync = ws.sync(repair=True, verify=True, progress=lambda r: print("repair", r, flush=True))
    sync_seconds = round(time.perf_counter() - t, 1)
    print(f"sync {sync_seconds}s: forgot {len(sync.get('forgotten', []))}, repaired {len(sync.get('repaired', []))}",
          flush=True)

    def rows(ids, scores):
        return [(cid, question[cid], scores(cid)) for cid in ids]

    edited = rows(edit, lambda cid: [("edited_new_value", new[cid]), ("edited_old_value_still_produced", old[cid])])
    after_sync = measure(ws, {
        "wiki": rows(edit, lambda cid: [("edited_new_value", new[cid])]),
        "engram": edited + rows(delete, lambda cid: [("deleted_still_produced", old[cid])]) +
        rows(keep, lambda cid: [("untouched_retained", claims[cid].object)]),
        "hybrid": edited + rows(delete, lambda cid: [("deleted_still_produced", old[cid])]) +
        rows(keep, lambda cid: [("untouched_retained", claims[cid].object)]),
    }, batch)
    print("after sync:", {k: v for k, v in after_sync.items() if k != "_detail"}, flush=True)

    t = time.perf_counter()
    recons = ws.consolidate(progress=lambda r: print("consolidate", r, flush=True))
    recons_seconds = round(time.perf_counter() - t, 1)
    after_recons = measure(ws, {"engram": rows(edit, lambda cid: [("edited_new_value", new[cid])]) +
                                rows(keep, lambda cid: [("untouched_retained", claims[cid].object)])}, batch)
    print("after re-consolidation:", {k: v for k, v in after_recons.items() if k != "_detail"}, flush=True)

    out = {
        "facts": len(held), "edited": len(edit), "deleted": len(delete), "untouched": len(keep),
        "edits": [{"claim": cid, "old": old[cid], "new": new[cid], "question": question[cid]} for cid in edit],
        "deletions": [{"claim": cid, "answer": old[cid], "question": question[cid]} for cid in delete],
        "sync": {"seconds": sync_seconds, "forgotten": len(sync.get("forgotten", [])),
                 "repaired": len(sync.get("repaired", [])), "verification": sync.get("verification", {})},
        "after_sync": {k: v for k, v in after_sync.items() if k != "_detail"},
        "reconsolidation": {"seconds": recons_seconds, "selected": recons.get("selected", 0),
                            "epochs": recons.get("training", {}).get("epochs_run"),
                            "verification": recons.get("verification", {})},
        "after_reconsolidation": {k: v for k, v in after_recons.items() if k != "_detail"},
    }
    return out, {"after_sync": after_sync["_detail"], "after_reconsolidation": after_recons["_detail"]}


def main() -> None:
    from wikiengram.runtime import WikiEngram

    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", default="update_results.json")
    ap.add_argument("--edit", type=int, default=15)
    ap.add_argument("--delete", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out, detail = run(WikiEngram.open(a.workspace), a.edit, a.delete, a.batch, a.seed)
    Path(a.out).write_text(json.dumps(out, indent=2))
    Path(a.out).with_name(Path(a.out).stem + "_detail.json").write_text(json.dumps(detail, indent=1,
                                                                                   ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("sync", "after_sync", "reconsolidation", "after_reconsolidation")},
                     indent=1))


if __name__ == "__main__":
    main()
