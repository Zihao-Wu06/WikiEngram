"""Training samples and verification probes built from wiki claims.

For each claim: a few question phrasings in QA format (target ``" <object>\\n"``) plus the statement as a
completion (``"X works at"`` -> ``" Y."``). One question phrasing is held out of training and used as a
verification probe, so verification measures recall beyond the exact trained wording.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..prompts import qa_prompt, statement_completion
from .trainer import TrainingSample


@dataclass
class Probe:
    claim_id: str
    prompt: str
    answer: str
    kind: str  # "trained_question" | "heldout_question" | "statement"


def claim_questions(claim, extractor=None, n: int = 3) -> list[str]:
    """Question phrasings for a claim: its stored questions, topped up by the extractor's paraphrases.
    An extractor with ``override_questions = True`` (e.g. the benchmark's fixed split) is used exclusively."""
    if extractor is not None and getattr(extractor, "override_questions", False):
        return list(extractor.paraphrase_questions(claim, n=n))[:n]
    qs = [q for q in (claim.questions or []) if q.strip()]
    if len(qs) < n and extractor is not None:
        for q in extractor.paraphrase_questions(claim, n=n + 1):
            if q not in qs:
                qs.append(q)
    return qs


def build_claim_samples(claim, questions: list[str], n_train_questions: int, include_statement: bool
                        ) -> tuple[list[TrainingSample], list[Probe]]:
    samples, probes = [], []
    train_qs = questions[:n_train_questions]
    held = questions[n_train_questions : n_train_questions + 1]
    for q in train_qs:
        samples.append(TrainingSample(claim.id, qa_prompt(q), f" {claim.object}\n"))
        probes.append(Probe(claim.id, qa_prompt(q), claim.object, "trained_question"))
    for q in held:
        probes.append(Probe(claim.id, qa_prompt(q), claim.object, "heldout_question"))
    if include_statement:
        split = statement_completion(claim.statement, claim.object)
        if split is not None:
            samples.append(TrainingSample(claim.id, split[0], split[1] + "\n"))
            probes.append(Probe(claim.id, split[0], claim.object, "statement"))
    if not samples:  # a claim with no question and an unsplittable statement: train the statement as-is
        samples.append(TrainingSample(claim.id, f"{claim.subject}:", f" {claim.statement}\n"))
    return samples, probes
