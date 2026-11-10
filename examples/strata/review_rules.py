"""First-pass review of extracted claims: flag what makes a poor memory, write ``fixes.json`` for ``build_memory.py
review``. Rules, in order:

* the answer is not a short value (longer than 60 characters);
* the answer is markup rather than knowledge: a Markdown link, inline code, a documentation file name, or a URL
  longer than 40 characters (short URLs such as a local API address are kept: they are useful facts);
* the answer is an identifier the extractor wrote instead of the value (snake_case such as ``28.8_gb_n_gram_table``):
  when the statement says the value in words ("... a 28.8 GB n-gram table on the SSD."), the answer is corrected
  to those words and the statement kept; when the statement only repeats the identifier, the claim is dropped;
* a duplicate of an earlier claim (same subject, predicate and answer after normalisation).

Claims without questions are kept: ``build_memory.py questions`` adds paraphrases. Everything flagged is printed
with its reason, so a person (or an agent) can check the list before applying it.

    python review_rules.py --claims WS/claims.json --out fixes.json
"""

from __future__ import annotations

import argparse
import json
import re

MARKUP = re.compile(r"`|\]\(|\[[^\]]+\]|\.md\b|\.py\b|\.json\b|\.sh\b|\.bat\b")
URL = re.compile(r"https?://\S+")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _span_in_statement(identifier: str, statement: str) -> str | None:
    """The words of ``statement`` that ``identifier`` stands for, unless the statement just repeats it."""
    if identifier.lower() in statement.lower():
        return None
    tokens = [t for t in re.split(r"[_\s-]+", identifier) if t]
    m = re.search(r"[\s_-]*".join(map(re.escape, tokens)), statement, re.IGNORECASE)
    return m.group(0).strip() if m else None


def review(claims: list[dict]) -> list[dict]:
    fixes, seen = [], set()
    for c in claims:
        obj = c["object"].strip()
        key = (_norm(c["subject"]), _norm(c["predicate"]), _norm(obj))
        reason = None
        if len(obj) > 60:
            reason = "answer too long to recall exactly"
        elif MARKUP.search(obj) or any(len(u) > 40 for u in URL.findall(obj)):
            reason = "answer is markup, not knowledge"
        elif "_" in obj and " " not in obj and not obj.isupper():
            span = _span_in_statement(obj, c["statement"])
            if span is None:
                reason = "answer is an invented identifier"
            else:
                seen.add(key)
                fixes.append({"id": c["id"], "action": "correct", "object": span, "statement": c["statement"],
                              "reason": f"review: answer written as {obj!r}, value in the statement is {span!r}"})
                continue
        elif key in seen:
            reason = "duplicate"
        seen.add(key)
        if reason:
            fixes.append({"id": c["id"], "action": "forget", "reason": f"review: {reason}",
                          "statement": c["statement"]})
    return fixes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.claims) as fh:
        claims = json.load(fh)
    fixes = review(claims)
    for f in fixes:
        print(f"{f['action']:8s} {f['reason'][:80]:80s} | {f['statement'][:70]}")
    with open(a.out, "w") as fh:
        json.dump([{k: v for k, v in f.items() if f["action"] == "correct" or k != "statement"} for f in fixes],
                  fh, indent=1)
    print(f"{len(fixes)} of {len(claims)} claims flagged -> {a.out}")


if __name__ == "__main__":
    main()
