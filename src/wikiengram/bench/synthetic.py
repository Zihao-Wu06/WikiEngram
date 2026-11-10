"""Deterministic fictitious facts for memory benchmarks.

Every entity and value is invented (generated from syllables and checked against a small stop-list), so a
pretrained model cannot know them in advance: whatever it answers correctly afterwards came from memory. Each
fact has *training* phrasings and *held-out* paraphrases that are never trained on, which measures whether the
memory generalizes beyond the exact wording (the hard case for n-gram memory).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..utils.text import answer_in

_ONSETS = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "j", "k", "kr", "l", "m", "n", "p", "qu", "r",
           "s", "st", "t", "th", "tr", "v", "vr", "w", "z", "zh"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "ia", "ou", "y"]
_CODAS = ["", "", "n", "r", "l", "s", "th", "x", "m", "nd", "rk", "sh"]


def _word(rng: random.Random, syllables: int) -> str:
    return "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS) for _ in range(syllables))


def _name(rng: random.Random) -> str:
    return f"{_word(rng, 2).capitalize()} {_word(rng, 2).capitalize()}"


RELATIONS: dict[str, dict] = {
    "works_at": {
        "value": lambda r: f"{_word(r, 2).capitalize()} {r.choice(['Labs', 'Industries', 'Systems', 'Foods', 'Media'])}",
        "statement": ["{s} works at {o}.", "{s} is employed by {o}.", "The employer of {s} is {o}."],
        "question": ["Where does {s} work?", "Which company employs {s}?", "What company does {s} work for?",
                     "Who is {s}'s employer?"],
    },
    "lives_in": {
        "value": lambda r: f"{r.choice(['Port', 'New', 'East', 'Lake', 'Fort'])} {_word(r, 2).capitalize()}",
        "statement": ["{s} lives in {o}.", "{s}'s home town is {o}.", "The city where {s} lives is {o}."],
        "question": ["Where does {s} live?", "In which city does {s} live?", "What is {s}'s home town?",
                     "Which town is {s} living in?"],
    },
    "pet_name": {
        "value": lambda r: _word(r, 2).capitalize(),
        "statement": ["{s}'s dog is named {o}.", "{s} has a dog called {o}.", "The name of {s}'s dog is {o}."],
        "question": ["What is the name of {s}'s dog?", "What is {s}'s dog called?", "What did {s} name the dog?",
                     "Whose name does {s}'s dog have? It is called"],
    },
    "favorite_dish": {
        "value": lambda r: f"{_word(r, 2)} {r.choice(['stew', 'pie', 'noodles', 'soup', 'curry'])}",
        "statement": ["{s}'s favorite dish is {o}.", "{s} loves eating {o}.", "The dish {s} likes best is {o}."],
        "question": ["What is {s}'s favorite dish?", "What food does {s} love most?",
                     "Which dish does {s} like best?", "What does {s} like to eat most?"],
    },
    "instrument": {
        "value": lambda r: f"the {_word(r, 2)}",
        "statement": ["{s} plays {o}.", "The instrument {s} plays is {o}.", "{s} is a player of {o}."],
        "question": ["What instrument does {s} play?", "Which instrument is played by {s}?",
                     "What does {s} play in the band?", "Which instrument can {s} play?"],
    },
}

QA_TEMPLATE = "Question: {q}\nAnswer:"


@dataclass
class Fact:
    fact_id: str
    subject: str
    relation: str
    obj: str
    train_statements: list[str] = field(default_factory=list)
    train_questions: list[str] = field(default_factory=list)
    heldout_questions: list[str] = field(default_factory=list)

    @property
    def statement(self) -> str:
        return self.train_statements[0]


def make_facts(n: int, seed: int = 0, train_questions: int = 2, train_statements: int = 2) -> list[Fact]:
    """``n`` facts with unique subjects. Questions beyond the first ``train_questions`` are held out."""
    rng = random.Random(seed)
    rels = list(RELATIONS)
    used: set[str] = set()
    facts = []
    for i in range(n):
        subj = _name(rng)
        while subj in used:
            subj = _name(rng)
        used.add(subj)
        rel = rels[i % len(rels)]
        spec = RELATIONS[rel]
        obj = spec["value"](rng)
        st = [t.format(s=subj, o=obj) for t in spec["statement"]]
        qs = [t.format(s=subj) for t in spec["question"]]
        facts.append(Fact(
            fact_id=f"f{i:04d}", subject=subj, relation=rel, obj=obj,
            train_statements=st[:train_statements],
            train_questions=qs[:train_questions],
            heldout_questions=qs[train_questions:],
        ))
    return facts


def statement_prefix(statement: str, obj: str) -> str:
    """'X works at Y.' -> 'X works at' (every template ends with the object)."""
    idx = statement.rfind(obj)
    return statement[:idx].rstrip()


def answer_matches(prediction: str, obj: str) -> bool:
    """True if the object appears in the generated answer as whole words (see :func:`answer_in`)."""
    return answer_in(prediction, obj)
