"""Verification probes: does Engram (alone, with no wiki context) recall or forget what it should?"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

from ..backends.hf import EngramCausalLM
from ..utils.text import answer_in
from .samples import Probe


def answer_contains(prediction: str, answer: str) -> bool:
    return answer_in(prediction, answer)


@dataclass
class ProbeResult:
    claim_id: str
    kind: str
    prompt: str
    expected: str
    prediction: str
    correct: bool


@dataclass
class VerificationReport:
    results: list[ProbeResult] = field(default_factory=list)

    def accuracy(self, kind: str | None = None) -> float:
        rs = [r for r in self.results if kind is None or r.kind == kind]
        return sum(r.correct for r in rs) / len(rs) if rs else float("nan")

    def per_claim(self) -> dict[str, bool]:
        """A claim is verified when every one of its probes is answered correctly."""
        acc: dict[str, list[bool]] = defaultdict(list)
        for r in self.results:
            acc[r.claim_id].append(r.correct)
        return {cid: all(v) for cid, v in acc.items()}

    def summary(self) -> dict:
        kinds = sorted({r.kind for r in self.results})
        return {"n": len(self.results), "accuracy": self.accuracy(),
                **{f"accuracy_{k}": self.accuracy(k) for k in kinds}}

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "results": [asdict(r) for r in self.results]}


def run_probes(lm: EngramCausalLM, probes: list[Probe], use_delta: bool = True, batch: int = 64,
               max_new_tokens: int = 32, expect_absent: bool = False) -> VerificationReport:
    """Generate for each probe with Engram on and no wiki context. With ``expect_absent`` a probe is
    "correct" when the answer is NOT produced (used to verify forgetting of deleted / outdated facts)."""
    report = VerificationReport()
    # statement probes continue the sentence as plain text, the way statements are trained (even for chat models)
    for raw in (False, True):
        group = [p for p in probes if (p.kind == "statement") == raw]
        for i in range(0, len(group), batch):
            chunk = group[i : i + batch]
            res = lm.generate([p.prompt for p in chunk], max_new_tokens=max_new_tokens, use_delta=use_delta,
                              raw=raw)
            _record(report, chunk, res.texts, expect_absent)
    return report


def _record(report: VerificationReport, chunk: list[Probe], texts: list[str], expect_absent: bool) -> None:
    for p, text in zip(chunk, texts):
        hit = answer_contains(text, p.answer)
        report.results.append(ProbeResult(p.claim_id, p.kind, p.prompt, p.answer, text,
                                          (not hit) if expect_absent else hit))
