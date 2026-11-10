"""Dual-layer memory inference.

Modes:

* ``none``   - the frozen backbone alone (no Engram, no wiki);
* ``wiki``   - LLMWiki only: retrieved claims go into the prompt (lexical retrieval, no vectors);
* ``engram`` - neural memory only: base Engram + personal delta, no context;
* ``hybrid`` - both, routed per question. If the most relevant retrieved claim is verifiably recalled by
  Engram at its current version, it is left out of the prompt and the personal delta answers it (that is
  where the token savings come from); only other unverified claims about the same subject stay in context,
  because unrelated facts in the prompt get copied as answers. Otherwise the wiki answers with the delta
  switched off, so new, edited or unverified knowledge always comes from the latest wiki state and memory
  cannot compete with it.

The personal delta is applied to the question tokens only, never to wiki context in the prompt (otherwise the
memories of every entity mentioned in the context fire and compete).

Before answering, pending wiki changes are applied to the neural memory instantly (rows of edited or deleted
claims are removed or masked, no training), so outdated memory never takes part in an answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..prompts import qa_prompt_parts

MODES = ("none", "wiki", "engram", "hybrid")


@dataclass
class Answer:
    question: str
    text: str
    mode: str
    context_claims: list[str] = field(default_factory=list)
    retrieved_claims: list[str] = field(default_factory=list)
    engram_claims: list[str] = field(default_factory=list)  # retrieved claims left to Engram (verified)
    prompt_tokens: int = 0
    new_tokens: int = 0
    seconds: float = 0.0
    first_token_seconds: float = 0.0
    answered_by: str = "wiki"

    def to_dict(self) -> dict:
        return asdict(self)


def is_verified_in_engram(ws, claim) -> bool:
    """Engram holds this claim at its current version, passed its probes, and none of its rows is masked
    waiting for repair."""
    info = ws.state.claims.get(claim.id)
    return bool(info and info.get("verified") and info.get("version") == claim.version and claim.id in ws.ledger
                and not ws.has_pending_repair(claim.id))


@dataclass
class Prepared:
    prompt: str
    context_tokens: int  # leading prompt tokens that are wiki context (the personal delta is off there)
    context: list
    retrieved: list
    engram: list
    use_engram: bool = True
    use_delta: bool = False
    answered_by: str = "wiki"  # "wiki" | "engram" | "backbone"


def prepare(ws, question: str, mode: str) -> Prepared:
    """Build the prompt for one question: retrieve claims, decide what goes into the context and whether the
    personal delta is used."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    hits = [c for c, _ in ws.wiki.search(question, k=ws.config.fusion.top_k)] if mode in ("wiki", "hybrid") else []
    context, engram = list(hits), []
    use_delta = mode == "engram"
    if mode == "hybrid":
        verified = [is_verified_in_engram(ws, c) for c in hits]
        # route: Engram answers when it verifiably holds the most relevant claim (or retrieval found nothing)
        use_delta = (not hits) or (verified[0] and ws.config.fusion.skip_verified_consolidated)
        if use_delta and hits:
            # the question goes to Engram alone: facts in the prompt get copied as answers, even facts about the
            # same subject (keeping those cost 8 of 149 trained-phrasing answers in the Strata example)
            engram = [c for c, v in zip(hits, verified) if v]
            context = []
        # on the wiki route the delta is off, so every retrieved claim (verified or not) stays in the prompt
    ctx, q = qa_prompt_parts(question, [c.statement for c in context], brief=ws.lm.prompt_format == "chat")
    # leading tokens of the prompt as the model sees it (special / chat-template tokens included) that are context
    n_ctx = ws.lm.context_tokens(ctx, q)
    answered_by = "backbone" if mode == "none" else ("engram" if use_delta else "wiki")
    return Prepared(ctx + q, n_ctx, context, hits, engram, mode != "none", use_delta, answered_by)


def answer_question(ws, question: str, mode: str, touch: bool = True) -> Answer:
    if mode in ("engram", "hybrid") and ws.pending_events():
        ws.sync(repair=False, verify=False)  # instant: stale rows removed / masked, no training
    p = prepare(ws, question, mode)
    delta_start = None if ws.config.fusion.delta_on_context else [p.context_tokens]
    res = ws.lm.generate([p.prompt], max_new_tokens=ws.config.fusion.max_new_tokens,
                         use_engram=p.use_engram, use_delta=p.use_delta, delta_start=delta_start)
    if touch and p.retrieved:
        ws.wiki.touch(sorted({p.retrieved[0].id, *[c.id for c in p.context], *[c.id for c in p.engram]}))
    return Answer(question, res.texts[0], mode, [c.id for c in p.context], [c.id for c in p.retrieved],
                  [c.id for c in p.engram], res.prompt_tokens[0], res.new_tokens[0], res.seconds,
                  res.first_token_seconds, p.answered_by)
