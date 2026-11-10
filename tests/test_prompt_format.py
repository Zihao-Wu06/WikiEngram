"""Prompt formats: chat and thinking models are asked through their own chat template (thinking off), base models
with plain text; training, verification and answering always agree on the format."""

from __future__ import annotations

import pytest

from tests.conftest import make_tiny_model, make_tiny_tokenizer
from tests.test_e2e_tiny import _with_base
from wikiengram.backends.hf import EngramCausalLM
from wikiengram.consolidation.trainer import DeltaTrainConfig, TrainingSample, encode_samples, train_delta
from wikiengram.prompts import qa_prompt, qa_prompt_parts, resolve_prompt_format

TEMPLATE = ("{% for m in messages %}<|user|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% if enable_thinking is defined and not enable_thinking %}"
            "<think></think>\n{% endif %}{% endif %}")


class _Tok:
    def __init__(self, template):
        self.chat_template = template


def test_resolve_prompt_format():
    chat_tok, plain_tok = _Tok(TEMPLATE), _Tok(None)
    assert resolve_prompt_format("auto", "/models/Qwen3.8-Flash-Next-FP8", chat_tok, native=True) == "chat"
    assert resolve_prompt_format("auto", "/models/Qwen3-1.7B-Base", chat_tok, native=True) == "completion"
    assert resolve_prompt_format("auto", "Qwen/Qwen3-0.6B-Base", chat_tok, native=False) == "completion"
    assert resolve_prompt_format("auto", "some/instruct-model", plain_tok, native=True) == "completion"
    assert resolve_prompt_format("chat", "x", plain_tok, native=False) == "chat"
    with pytest.raises(ValueError):
        resolve_prompt_format("sometimes", "x", chat_tok, native=True)


@pytest.fixture()
def chat_lm(tiny_spec):
    tok = make_tiny_tokenizer()
    tok.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|end|>", "<|assistant|>", "<think>",
                                                          "</think>"]})
    tok.chat_template = TEMPLATE
    lm = EngramCausalLM(make_tiny_model(len(tok)), tok, tiny_spec, device="cpu", name="tiny-chat")
    lm.prompt_format = "chat"
    return _with_base(lm)


def test_chat_format_wraps_questions_and_counts_context(chat_lm):
    lm = chat_lm
    q = qa_prompt("Where does Zorblat Quenn work ?")
    assert lm.format_prompt(q) == f"<|user|>\n{q}<|end|>\n<|assistant|>\n<think></think>\n"
    before, after = lm.prompt_wrapper()
    assert before == "<|user|>\n" and after == "<|end|>\n<|assistant|>\n<think></think>\n"
    ctx, qq = qa_prompt_parts("Where does Zorblat Quenn work ?", ["Zorblat Quenn works at Vexmora Labs ."])
    n = lm.context_tokens(ctx, qq)
    full = lm.tokenizer(lm.format_prompt(ctx + qq), add_special_tokens=False)["input_ids"]
    head = lm.tokenizer("<|user|>\n" + ctx, add_special_tokens=False)["input_ids"]
    assert n == len(head) and full[:n] == head
    # question-answer samples are wrapped and their answer starts a fresh line; statement completions stay plain
    qa = TrainingSample("c1", q, " Vexmora Labs\n", qa=True)
    plain = TrainingSample("c1", "Zorblat Quenn works at", " Vexmora Labs .\n")
    (ids, start), (pids, pstart) = encode_samples(lm.tokenizer, [qa, plain], lm)
    assert ids[:start] == lm.tokenizer(lm.format_prompt(q), add_special_tokens=False)["input_ids"]
    # the answer ends the assistant turn (end-of-turn token), so a chat client gets the answer and nothing more
    assert ids[start:] == lm.tokenizer("Vexmora Labs" + lm.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    assert pids[:pstart] == lm.tokenizer(plain.prompt)["input_ids"]


def test_memory_trained_in_chat_format_is_recalled_when_asked_in_chat_format(chat_lm):
    lm = chat_lm
    q1, q2 = qa_prompt("Where does Zorblat Quenn work ?"), qa_prompt("Where does Mirel Kesh live ?")
    samples = [TrainingSample("c1", q1, " Vexmora Labs\n", qa=True),
               TrainingSample("c1", "Zorblat Quenn works at", " Vexmora Labs .\n"),
               TrainingSample("c2", q2, " Port Elrin\n", qa=True),
               TrainingSample("c2", "Mirel Kesh lives in", " Port Elrin .\n")]
    rep = train_delta(lm, lm.new_delta(), samples, DeltaTrainConfig(epochs=200, batch_size=8, lr=5e-2))
    assert rep.stop_reason in ("exact", "target_loss")
    out = lm.generate([q1, q2], max_new_tokens=6).texts  # generate applies the same chat wrapping
    assert out == ["Vexmora Labs", "Port Elrin"]
    # statements are trained as plain text continuations, so they are verified that way too (no chat wrapping)
    from wikiengram.consolidation.samples import Probe
    from wikiengram.consolidation.verify import run_probes

    probes = [Probe("c1", q1, "Vexmora Labs", "trained_question"),
              Probe("c1", "Zorblat Quenn works at", "Vexmora Labs", "statement"),
              Probe("c2", "Mirel Kesh lives in", "Port Elrin", "statement")]
    report = run_probes(lm, probes, max_new_tokens=6)
    assert report.accuracy("statement") == 1.0 and report.accuracy("trained_question") == 1.0


def test_chat_models_also_train_plain_questions(chat_lm):
    from wikiengram.consolidation.samples import build_claim_samples
    from wikiengram.wiki.models import Claim

    claim = Claim(id="c1", subject="Zorblat Quenn", predicate="works_at", object="Vexmora Labs",
                  statement="Zorblat Quenn works at Vexmora Labs .")
    q = "Where does Zorblat Quenn work ?"
    plain, _ = build_claim_samples(claim, [q], 1, False)
    both, _ = build_claim_samples(claim, [q], 1, False, bare_questions=True)
    assert [s.prompt for s in plain] == [qa_prompt(q)]
    assert [s.prompt for s in both] == [qa_prompt(q), q] and all(s.qa for s in both)
    # the plain question is wrapped exactly as the chat API wraps a user message
    (ids, start), = encode_samples(chat_lm.tokenizer, [both[1]], chat_lm)
    assert ids[:start] == chat_lm.tokenizer(chat_lm.format_prompt(q), add_special_tokens=False)["input_ids"]


def test_model_api_asks_base_models_in_the_trained_format():
    from wikiengram.api.openai_server import render

    text, special = render(_Tok(None), [{"role": "user", "content": "Where does Zorblat Quenn work ?"}], None, False)
    assert text == qa_prompt("Where does Zorblat Quenn work ?") and special
    text, _ = render(_Tok(None), [{"role": "system", "content": "Be brief."},
                                  {"role": "user", "content": "Where does Zorblat Quenn work ?"}], None, False)
    assert text == "System: Be brief.\n" + qa_prompt("Where does Zorblat Quenn work ?")
