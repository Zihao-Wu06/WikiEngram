"""Claim extraction: turn text into atomic ``(subject, predicate, object)`` claims.

``RuleExtractor`` is deterministic and offline (tests, CI, the synthetic benchmark). ``OpenAICompatExtractor``
calls any OpenAI-compatible ``/chat/completions`` endpoint (DeepSeek API, vLLM, llama.cpp, Ollama, ...).
Facts a user states about themselves get the subject ``"User"`` ("The user lives in Port Elrin.").
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

import httpx

from .models import Claim, is_cjk, normalize, snake_case

log = logging.getLogger(__name__)
T = TypeVar("T")

USER = "User"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class ExtractedClaim:
    """A claim proposed by an extractor, before reconciliation with the wiki."""

    subject: str
    predicate: str
    object: str
    statement: str
    confidence: float = 0.8
    importance: float = 0.5
    multi_valued: bool = False
    questions: list[str] = field(default_factory=list)
    category: str = "fact"
    scope: str = ""


@runtime_checkable
class Extractor(Protocol):
    def extract(self, text: str, kind: str = "document") -> list[ExtractedClaim]:
        """Atomic claims stated in ``text``; ``kind`` is "document", "conversation" or "note"."""
        ...

    def paraphrase_questions(self, claim: Claim, n: int = 3) -> list[str]:
        """Natural question phrasings whose answer is ``claim.object``."""
        ...


# ------------------------------------------------------------------------------------------ text helpers
_TRAILING_PUNCT = ".,;:!?。，、；：！？…"
_QUOTES = "\"'“”‘’«»「」『』"
_TERMINALS = ".!?。！？"

# Synonymous attribute names folded into the canonical predicates (keeps conflict detection consistent).
_ALIASES = {
    "employer": "works_at",
    "workplace": "works_at",
    "home_town": "lives_in",
    "hometown": "lives_in",
    "home_city": "lives_in",
    "city_of_residence": "lives_in",
    "place_of_residence": "lives_in",
    "residence": "lives_in",
    "dog": "pet_name",
    "dog_name": "pet_name",
    "favourite_dish": "favorite_dish",
    "birthplace": "birth_place",
    "place_of_birth": "birth_place",
    "year_of_birth": "birth_year",
    "date_of_birth": "birth_date",
}

# Canonical statements; each ends with the object ({s}: subject phrase, {p}: its possessive).
_TEMPLATES = {
    "works_at": "{s} works at {o}.",
    "lives_in": "{s} lives in {o}.",
    "pet_name": "{p} dog is named {o}.",
    "favorite_dish": "{p} favorite dish is {o}.",
    "instrument": "{s} plays {o}.",
    "plays": "{s} plays {o}.",
    "likes": "{s} likes {o}.",
    "birth_year": "{s} was born in {o}.",
    "birth_place": "{s} was born in {o}.",
    "birth_date": "{s} was born on {o}.",
}

_QUESTION_TEMPLATES = {
    "works_at": ["Where does {s} work?", "Which company employs {s}?", "At which company does {s} work?",
                 "Who employs {s}?"],
    "lives_in": ["Where does {s} live?", "In which city does {s} live?", "Where is {p} home?",
                 "In what place does {s} live?"],
    "pet_name": ["What is the name of {p} dog?", "What is {p} dog called?", "What name does {p} dog have?"],
    "favorite_dish": ["What is {p} favorite dish?", "What food does {s} love most?",
                      "Which meal does {s} prefer above all others?"],
    "instrument": ["What instrument does {s} play?", "Which instrument is played by {s}?",
                   "On which instrument does {s} perform?"],
    "plays": ["What does {s} play?", "Which game does {s} play?"],
    "likes": ["What does {s} like?", "What does {s} enjoy?", "What is something {s} likes?"],
    "birth_year": ["In what year was {s} born?", "When was {s} born?", "What is {p} year of birth?"],
    "birth_place": ["Where was {s} born?", "What is {p} birthplace?", "In which place was {s} born?"],
    "birth_date": ["When was {s} born?", "What is {p} date of birth?"],
}


def clean_value(text: str) -> str:
    """Collapse whitespace and strip wrapping quotes and trailing punctuation ("'Brix'." -> "Brix")."""
    s = " ".join(str(text).split())
    while True:
        prev = s
        s = s.rstrip(_TRAILING_PUNCT).strip()
        if len(s) >= 2 and s[0] in _QUOTES and s[-1] in _QUOTES:
            s = s[1:-1].strip()
        if s == prev:
            return s


def finish_sentence(text: str) -> str:
    """One clean sentence: collapsed whitespace, terminal punctuation, capitalized first word."""
    s = " ".join(str(text).split())
    if not s:
        return ""
    if s[-1] not in _TERMINALS:
        s += "。" if is_cjk(s[-1]) else "."
    first = s.split(" ", 1)[0]
    if first[:1].islower() and first == first.lower():
        s = s[0].upper() + s[1:]
    return s


def canonical_predicate(attr: str) -> str:
    """snake_case an attribute name and fold known synonyms ("home town" -> "lives_in")."""
    pred = snake_case(attr)
    return _ALIASES.get(pred, pred)


def subject_phrase(subject: str, start: bool = True) -> str:
    """How a subject reads in a sentence: ``"User"`` -> "The user", ``"User's sister"`` -> "The user's sister"."""
    if subject == USER or subject.startswith(USER + "'s "):
        phrase = "the user" + subject[len(USER) :]
        return phrase[0].upper() + phrase[1:] if start else phrase
    return subject


def possessive(phrase: str) -> str:
    return phrase + "'s"


def attr_words(attr: str) -> str:
    """Attribute as written in a sentence: capitalized words lowered, acronyms kept ("Home Town" -> "home town")."""
    words = attr.split()
    return " ".join(w.lower() if w[:1].isupper() and w[1:] == w[1:].lower() else w for w in words)


def make_statement(subject: str, predicate: str, obj: str, attr: str | None = None) -> str:
    """Canonical sentence for a claim; it always ends with the object."""
    s = subject_phrase(subject)
    p = possessive(s)
    template = _TEMPLATES.get(predicate)
    if template:
        return finish_sentence(template.format(s=s, p=p, o=obj))
    if predicate.endswith("_name") and predicate.count("_") == 1:
        return finish_sentence(f"{p} {predicate[:-5]} is named {obj}.")
    return finish_sentence(f"{p} {attr or predicate.replace('_', ' ')} is {obj}")


def template_questions(subject: str, predicate: str, n: int = 3) -> list[str]:
    """Template questions whose answer is the claim's object."""
    s = subject_phrase(subject, start=False)
    p = possessive(s)
    templates = _QUESTION_TEMPLATES.get(predicate)
    if templates is None:
        if predicate.endswith("_name") and predicate.count("_") == 1:
            animal = predicate[:-5]
            templates = [f"What is the name of {{p}} {animal}?", f"What is {{p}} {animal} called?"]
        else:
            words = predicate.replace("_", " ")
            templates = [f"What is {{p}} {words}?", f"What is the {words} of {{s}}?", f"Do you know {{p}} {words}?"]
    return [t.format(s=s, p=p) for t in templates][: max(n, 0)]


def _clamp(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if v != v else min(1.0, max(0.0, v))  # v != v: NaN


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def contains_value(text: str, value: str) -> bool:
    """True if ``value`` occurs in ``text`` as whole words (comparison forms; substring for CJK)."""
    t, v = normalize(text), normalize(value)
    if not v:
        return False
    if any(is_cjk(ch) for ch in v):
        return v in t
    return f" {v} " in f" {t} "


def _clean_questions(questions: Any, obj: str, limit: int = 5) -> list[str]:
    """Keep distinct non-empty question strings that do not give the answer away."""
    if not isinstance(questions, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if not isinstance(q, str):
            continue
        q = " ".join(q.split())
        norm = normalize(q)
        if not norm or norm in seen or contains_value(q, obj):
            continue
        if q[-1] not in "?？":
            q += "？" if is_cjk(q[-1]) else "?"
        seen.add(norm)
        out.append(q)
    return out[:limit]


def prepare_claims(items: Iterable[ExtractedClaim]) -> list[ExtractedClaim]:
    """Validate and canonicalize extractor output: clean fields, snake_case predicates, statements that
    contain the object, clamped scores, and no duplicate (subject, predicate, object) triples."""
    out: list[ExtractedClaim] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        subject = clean_value(item.subject)
        predicate = snake_case(item.predicate)
        obj = clean_value(item.object)
        if not (subject and predicate and obj):
            continue
        triple = (normalize(subject), normalize(predicate), normalize(obj))
        if triple in seen:
            continue
        seen.add(triple)
        statement = finish_sentence(item.statement or "")
        if not statement or not contains_value(statement, obj):
            statement = make_statement(subject, predicate, obj)
        out.append(ExtractedClaim(
            subject=subject, predicate=predicate, object=obj, statement=statement,
            confidence=_clamp(item.confidence, 0.8), importance=_clamp(item.importance, 0.5),
            multi_valued=_as_bool(item.multi_valued), questions=_clean_questions(item.questions, obj, 10),
        ))
    return out


# ----------------------------------------------------------------------------------------- sentences
_TITLES = frozenset({
    "mr", "mrs", "ms", "mx", "dr", "prof", "sr", "jr", "st", "mt", "rev", "gen", "col", "capt", "lt", "sgt",
    "hon", "fr",
})
_ENDING_ABBREV = frozenset({"etc", "inc", "ltd", "co", "corp", "llc", "plc"})  # may end a sentence
_INNER_ABBREV = frozenset({
    "vs", "cf", "approx", "no", "nos", "fig", "e.g", "i.e", "a.m", "p.m", "ph.d", "u.s", "u.k", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
})
_SENT_END = re.compile(r"[.!?。！？]+")
_LAST_WORD = re.compile(r"([\w.&#@+'-]+)$")  # "AT&T." and "C#." are words, not initials


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking at "Dr.", "e.g.", initials or decimal numbers."""
    out: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        if m.group()[-1] not in "。！？":
            nxt = text[end : end + 1]
            if nxt and not (nxt.isspace() or nxt in "\"'”’)]"):
                continue
            if m.group() == ".":
                word = _LAST_WORD.search(text[start : m.start()])
                w = word.group(1).casefold() if word else ""
                if (len(w) == 1 and w.isalpha()) or w in _TITLES or w in _INNER_ABBREV:
                    continue
                if w in _ENDING_ABBREV and not text[end:].lstrip()[:1].isupper():
                    continue
        sentence = text[start:end].strip()
        if sentence:
            out.append(sentence)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


# ------------------------------------------------------------------------------------------- patterns
_S = r"(?P<s>.+?)"
_O = r"(?P<o>.+)"
_POSS = r"['’]s"
_PETS = r"cat|rabbit|horse|parrot|bird|hamster|fish|turtle|snake|pony|ferret"
_MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"

# (pattern, handler); the first pattern whose handler yields claims wins. Order matters: specific forms
# ("The dish X likes best is Y") must precede the generic verbs and possessives that would also match.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.I), h)
    for p, h in [
        (rf"the employer of {_S} is {_O}", "works_at"),
        (rf"the (?:city|town|place|village) (?:where|in which) {_S} lives is {_O}", "lives_in"),
        (rf"the name of {_S}{_POSS} dog is {_O}", "pet_name"),
        (rf"the (?:dish|food|meal) {_S} (?:likes|loves|enjoys) (?:best|most) is {_O}", "favorite_dish"),
        (rf"the instrument {_S} plays is {_O}", "instrument"),
        (rf"{_S}{_POSS} dog is (?:named|called) {_O}", "pet_name"),
        (rf"{_S} has a dog (?:named|called) {_O}", "pet_name"),
        (rf"{_S}{_POSS} (?P<pet>{_PETS}) is (?:named|called) {_O}", "pet"),
        (rf"{_S} has an? (?P<pet>{_PETS}) (?:named|called) {_O}", "pet"),
        (rf"{_S} (?:works|is working) (?:at|for) {_O}", "works_at"),
        (rf"{_S} is employed (?:by|at) {_O}", "works_at"),
        (rf"{_S} (?:lives|resides|is living) in {_O}", "lives_in"),
        (rf"{_S} loves eating {_O}", "favorite_dish"),
        (rf"{_S} is a player of {_O}", "instrument"),
        (rf"{_S} plays {_O}", "plays"),
        (rf"{_S} was born in (?P<place>.+?),? in (?P<year>\d{{4}})", "born"),
        (rf"{_S} was born in (?:(?:early|mid|late)[- ])?(?:{_MONTHS}),? (?P<o>\d{{4}})", "birth_year"),
        (rf"{_S} was born in (?P<o>\d{{4}})", "birth_year"),
        (rf"{_S} was born on {_O}", "birth_date"),
        (rf"{_S} was born in {_O}", "birth_place"),
        (rf"{_S} (?:likes|enjoys|loves) {_O}", "likes"),
        (rf"{_S}{_POSS} (?P<attr>[^\W\d_][\w'’ -]*?) (?P<verb>is|are|was|were) (?:called |named )?{_O}",
         "possessive"),
        (rf"the (?P<attr>[^\W\d_][\w -]*?) of {_S} (?P<verb>is|are|was|were) {_O}", "of"),
    ]
]

# "I work at X and my favorite dish is Y": split at "and" when a new subject starts a clause
_CLAUSE_SPLIT = re.compile(r",?\s+and\s+(?=(?:my|I|I'm|he|she|his|her|they|their|we|our)\b)")
_MAX_SENTENCE = 600  # longer "sentences" are not atomic facts (and would make the patterns slow)
_NEGATION = re.compile(r"\b(?:not|never|nobody|cannot|no longer)\b|n['’]t\b", re.I)
_LEAD = re.compile(
    r"^(?:(?:oh|ok|okay|well|so|and|but|also|btw|by the way|fyi|actually|anyway|in fact|yes|yeah|no|hey|hi|"
    r"hello|note that|please note that|remember that|remember|just so you know|for the record|now|today|"
    r"currently|nowadays|these days|recently)\b[,:!]?\s*)+",
    re.I,
)
_HEDGE_LEAD = re.compile(
    r"^(?:i think|i believe|i guess|i suppose|i heard|apparently|reportedly|maybe|perhaps|probably|possibly|"
    r"presumably|supposedly|allegedly)\b(?:\s+that\b)?[,\s]*",
    re.I,
)
_HEDGE_INNER = re.compile(  # lowercase only: "Maybe Labs" is a name
    r"\s+(?:probably|maybe|perhaps|possibly|apparently|reportedly|presumably|supposedly|allegedly)\b"
)
_FIRST_PERSON = [
    (re.compile(r"^my\s+", re.I), "User's "),
    (re.compile(r"^i(?:['’]m| am)\s+", re.I), "User is "),
    (re.compile(r"^i was\s+", re.I), "User was "),
    (re.compile(r"^i(?:['’]ve| have)\s+", re.I), "User has "),
]
_FIRST_PERSON_VERB = re.compile(
    r"^i\s+(?:(?:also|still|now|currently|really|actually|usually|recently|mostly|often)\s+)*(?P<verb>[a-z]+)\b",
    re.I,
)
_THIRD_PERSON_PRONOUN = re.compile(r"^(?:he|she)\s+", re.I)
_POSSESSIVE_PRONOUN = re.compile(r"^(?:his|her)\s+", re.I)

_PRONOUNS = frozenset({
    "i", "me", "you", "he", "she", "it", "we", "they", "him", "her", "us", "them", "this", "that", "these",
    "those", "there", "here", "who", "what", "which", "someone", "somebody", "everyone", "everybody",
    "nobody", "anyone", "anybody", "one", "his", "hers", "its", "our", "their", "your", "my", "mine",
})
_LEAD_STOP = frozenset({
    "if", "when", "while", "because", "although", "though", "unless", "whether", "and", "but", "or", "so",
    "then", "also", "not", "no", "every", "each", "all", "some", "any", "many", "most", "few", "what",
    "which", "who", "where", "why", "how", "a", "an",
})
_SUBJECT_TRAILING = frozenset({
    "also", "still", "now", "currently", "really", "actually", "usually", "recently", "presently", "already",
    "mostly", "often", "sometimes", "too", "probably", "maybe", "perhaps", "apparently", "reportedly",
    "possibly", "presumably", "supposedly",
})
_DETERMINERS = frozenset({"the", "this", "that", "these", "those", "its", "our", "their", "his", "her"})
_SUBJECT_BAD = re.compile(
    r"\b(?:is|are|was|were|be|been|where|which|who|whom|whose|that|when|if|because|while|although|whether)\b",
    re.I,
)
_ATTR_BAD = re.compile(r"\b(?:and|or|but|that|which|who|whom|whose|when|where|because|if|not|has|have|had)\b", re.I)
_OBJECT_CUT = re.compile(
    r"(?:,\s*|\s+)(?:because|since|where|which|who|whom|whose|but|although|though|when|while|until|after|before|"
    r"so that|according to)\b.*$|,\s*(?:and|so|or)\s.*$",
    re.I,
)
_OBJECT_TRAILING = re.compile(
    r"(?:,?\s+(?:now|currently|these days|nowadays|at the moment|at present|too|as well|again|anymore|today|"
    r"already|still))+$",
    re.I,
)
_PARENTHESIS = re.compile(r"\s*\([^)]*\)\s*$")
_NOT_INSTRUMENT = re.compile(r"\b(?:role|part|game|games|music|songs?|tricks?|along|with|against)\b", re.I)
_GAMES = frozenset({
    "chess", "tennis", "football", "soccer", "basketball", "baseball", "golf", "cricket", "rugby", "hockey",
    "volleyball", "badminton", "poker", "bridge", "go", "squash", "darts", "snooker", "billiards", "pool",
    "handball", "lacrosse", "softball",
}) | {"table tennis", "ping pong", "video games", "cards", "board games"}


def _conjugate(verb: str) -> str:
    """Third person singular of an English verb in the present tense."""
    v = verb.lower()
    special = {"have": "has", "do": "does", "go": "goes", "am": "is", "was": "was"}
    if v in special:
        return special[v]
    if v.endswith(("s", "sh", "ch", "x", "z", "o")):
        return v + "es"
    if len(v) > 1 and v.endswith("y") and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _third_person(sentence: str) -> str:
    """Rewrite a first-person sentence about the user in the third person ("I live in X" -> "User lives in X")."""
    for pattern, replacement in _FIRST_PERSON:
        if pattern.match(sentence):
            return pattern.sub(replacement, sentence, count=1)
    m = _FIRST_PERSON_VERB.match(sentence)
    if m:
        return f"{USER} {_conjugate(m.group('verb'))}{sentence[m.end():]}"
    return sentence


def _clean_subject(raw: str) -> str | None:
    words = clean_value(raw).split()
    while words and words[-1].casefold() in _SUBJECT_TRAILING:
        words.pop()
    s = " ".join(words)
    low = s.casefold()
    if low == "the user":
        return USER
    if low.startswith("the user's "):
        s, words = f"{USER}'s {s[11:]}", [USER, *words[2:]]
    if not s or len(words) > 8 or low in _PRONOUNS or words[0].casefold() in _LEAD_STOP:
        return None
    if _SUBJECT_BAD.search(s):
        return None
    # Named entities only: an uppercase letter (beyond a sentence-initial determiner), a digit or a non-Latin
    # script. Rejects "The dog lives in the garden."
    core = " ".join(words[1:]) if words[0].casefold() in _DETERMINERS and len(words) > 1 else s
    if not any(ch.isupper() or ch.isdigit() or (ch.isalpha() and not ch.isascii()) for ch in core):
        return None
    return s


def _clean_object(raw: str) -> str | None:
    o = clean_value(raw)
    o = _OBJECT_CUT.sub("", o)
    o = _PARENTHESIS.sub("", o)
    o = clean_value(_OBJECT_TRAILING.sub("", o))
    words = o.split()
    if not words or len(words) > 12 or o.casefold() in _PRONOUNS or not any(ch.isalnum() for ch in o):
        return None
    return o


_MD_SUBS = [
    (re.compile(r"~~.*?~~"), ""),  # struck-through text is not asserted
    (re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]"), r"\2"),
    (re.compile(r"\[\[([^\]]+)\]\]"), r"\1"),
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)"), r"\1"),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"<[^>]+>"), ""),
]
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(?:```|~~~)")
_KEY_VALUE = re.compile(r"^(?P<key>[^:：]{1,60}?)\s*(?<!\d)[:：]\s*(?P<value>\S.*)$")
_GENERIC_SECTIONS = frozenset({
    "notes", "note", "facts", "fact", "details", "detail", "info", "information", "profile", "overview",
    "summary", "background", "contact", "contacts", "personal", "about", "misc", "miscellaneous", "people",
    "general", "bio", "biography", "basics", "todo", "links", "references", "sources", "history", "claims",
    "related", "index", "log",
}) | {"key facts", "quick facts", "see also", "fun facts", "personal details", "personal information"}
_SELF_SECTIONS = frozenset({"me", "about me", "myself", "user", "the user", "about the user"})
_META_KEYS = frozenset({
    "note", "notes", "summary", "description", "comment", "comments", "todo", "tags", "tag", "source",
    "sources", "related", "links", "tldr", "title", "date", "updated", "created", "status",
}) | {"see also"}


def _strip_markdown(text: str) -> str:
    for pattern, replacement in _MD_SUBS:
        text = pattern.sub(replacement, text)
    return text


class RuleExtractor:
    """Deterministic sentence-pattern extractor (offline; used by tests, CI and the synthetic benchmark).

    Recognizes the relations of the synthetic benchmark (works_at, lives_in, pet_name, favorite_dish,
    instrument), likes (multi-valued), birth year/place/date, generic "X's <attr> is Y" / "The <attr> of X is Y",
    and "Key: value" lines under a markdown heading that names the subject. Hedged sentences ("probably",
    "I think") get ``hedged_confidence``; negated sentences and questions are skipped. For kind "conversation"
    and "note", first person refers to the user.
    """

    def __init__(self, confidence: float = 0.8, hedged_confidence: float = 0.5, questions_per_claim: int = 3) -> None:
        self.confidence = confidence
        self.hedged_confidence = hedged_confidence
        self.questions_per_claim = questions_per_claim

    def __repr__(self) -> str:
        return f"RuleExtractor(confidence={self.confidence}, hedged_confidence={self.hedged_confidence})"

    def paraphrase_questions(self, claim: Claim, n: int = 3) -> list[str]:
        return template_questions(claim.subject, claim.predicate, n)

    def extract(self, text: str, kind: str = "document") -> list[ExtractedClaim]:
        first_person = kind in ("conversation", "note")
        out: list[ExtractedClaim] = []
        headings: list[tuple[int, str]] = []
        paragraph: list[str] = []
        in_fence = False

        def section_subject() -> str | None:
            for _, title in reversed(headings):
                norm = normalize(title)
                if norm in _SELF_SECTIONS:
                    return USER
                if norm and norm not in _GENERIC_SECTIONS and len(title.split()) <= 6 and not title.endswith("?"):
                    return title
            return None

        def flush() -> None:
            if paragraph:
                out.extend(self._prose(" ".join(paragraph), section_subject(), first_person))
                paragraph.clear()

        for line in text.splitlines():
            if _FENCE.match(line):
                flush()
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if not line.strip():
                flush()
                continue
            heading = _HEADING.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, clean_value(_strip_markdown(heading.group(2)))))
                continue
            bullet = _BULLET.match(line)
            body = _strip_markdown(line[bullet.end() :] if bullet else line).strip()
            subject = section_subject()
            labelled = self._labelled(body, subject, first_person)
            if labelled is not None:
                flush()
                out.extend(labelled)
            elif bullet:
                flush()
                out.extend(self._prose(body, subject, first_person))
            else:
                paragraph.append(body)
        flush()
        return prepare_claims(out)

    # ----------------------------------------------------------------------------------------- internals
    def _claim(self, subject: str, predicate: str, obj: str, confidence: float, statement: str | None = None,
               multi: bool = False) -> ExtractedClaim:
        return ExtractedClaim(
            subject=subject, predicate=predicate, object=obj,
            statement=finish_sentence(statement or make_statement(subject, predicate, obj)),
            confidence=confidence, multi_valued=multi,
            questions=template_questions(subject, predicate, self.questions_per_claim),
        )

    def _labelled(self, body: str, subject: str | None, first_person: bool) -> list[ExtractedClaim] | None:
        """Handle a "Label: value" line; ``None`` if the line is not one.

        A value that is a sentence of its own ("Update: she lives in X.") is read as that sentence; otherwise,
        under a heading naming the subject, the line is an attribute ("Employer: Vexmora Labs").
        """
        m = _KEY_VALUE.match(body)
        if not m:
            return None
        key, value = clean_value(m.group("key")), clean_value(m.group("value"))
        if not key or not value or value.startswith("//") or not key[0].isalpha():
            return None
        if len(key.split()) > 5 or any(ch in key for ch in ".?!"):
            return None
        sentence = self._prose(m.group("value"), subject, first_person)
        if sentence or subject is None or normalize(key) in _META_KEYS or len(value.split()) > 12:
            return sentence
        predicate = canonical_predicate(key)
        statement = make_statement(subject, predicate, value, attr=attr_words(key))
        return [self._claim(subject, predicate, value, self.confidence, statement=statement)]

    def _prose(self, text: str, section: str | None, first_person: bool) -> list[ExtractedClaim]:
        return [
            claim
            for part in re.split(r";\s+", text)
            for sentence in split_sentences(part)
            for clause in _CLAUSE_SPLIT.split(sentence)
            for claim in self._sentence(clause, section, first_person)
        ]

    def _sentence(self, sentence: str, section: str | None, first_person: bool) -> list[ExtractedClaim]:
        s = " ".join(sentence.split())
        if not s or len(s) > _MAX_SENTENCE or s.endswith(("?", "？")) or _NEGATION.search(s):
            return []
        s = s.rstrip(_TRAILING_PUNCT + " ")
        confidence = self.confidence
        s = _LEAD.sub("", s)
        hedged = _HEDGE_LEAD.sub("", s)
        hedged = _LEAD.sub("", _HEDGE_INNER.sub("", hedged, count=1))
        if hedged != s:
            s, confidence = hedged, self.hedged_confidence
        if first_person:
            s = _third_person(s)
        if section:
            s = _THIRD_PERSON_PRONOUN.sub(lambda _: section + " ", s, count=1)
            s = _POSSESSIVE_PRONOUN.sub(lambda _: possessive(section) + " ", s, count=1)
        for pattern, handler in _RULES:
            m = pattern.fullmatch(s)
            if m:
                claims = self._apply(handler, m, confidence)
                if claims:
                    return claims
        return []

    def _apply(self, handler: str, m: re.Match[str], confidence: float) -> list[ExtractedClaim]:
        groups = m.groupdict()
        subject = _clean_subject(groups["s"])
        if subject is None:
            return []
        if handler == "born":
            place = _clean_object(groups["place"])
            if place is None:
                return []
            return [self._claim(subject, "birth_place", place, confidence),
                    self._claim(subject, "birth_year", groups["year"], confidence)]
        obj = _clean_object(groups["o"])
        if obj is None:
            return []
        if handler == "plays":
            if normalize(re.sub(r"^the\s+", "", obj, flags=re.I)) in _GAMES:
                return [self._claim(subject, "plays", obj, confidence, multi=True)]
            if len(obj.split()) > 5 or _NOT_INSTRUMENT.search(obj):
                return []
            handler = "instrument"
        if handler == "pet":
            pet = groups["pet"].lower()
            return [self._claim(subject, f"{pet}_name", obj, confidence)]
        if handler in ("possessive", "of"):
            attr = groups["attr"].strip()
            if not 1 <= len(attr.split()) <= 4 or _ATTR_BAD.search(attr):
                return []
            predicate = canonical_predicate(attr)
            if not predicate:
                return []
            if predicate in _TEMPLATES:
                return [self._claim(subject, predicate, obj, confidence)]
            words, verb = attr_words(attr), groups["verb"].lower()
            if handler == "possessive":
                statement = f"{possessive(subject_phrase(subject))} {words} {verb} {obj}."
            else:
                statement = f"The {words} of {subject_phrase(subject, start=False)} {verb} {obj}."
            return [self._claim(subject, predicate, obj, confidence, statement=statement)]
        if handler == "birth_year" and not re.fullmatch(r"\d{4}", obj):
            return []
        return [self._claim(subject, handler, obj, confidence, multi=handler == "likes")]


# --------------------------------------------------------------------------------- OpenAI-compatible LLM
class LLMError(RuntimeError):
    """An LLM request failed (HTTP error, timeout or unusable output). Messages never contain the API key."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_EXTRACT_PROMPT = """You turn text into atomic facts for a personal knowledge base.

Reply with one JSON object and nothing else:
{"claims": [{"subject": "...", "predicate": "...", "object": "...", "statement": "...", "confidence": 0.9, \
"importance": 0.5, "multi_valued": false, "questions": ["..."]}]}

Rules:
- One fact per claim. "subject" is the entity the fact is about, with its full name as written in the text. \
"object" is a short value: a name, place, date, number or brief phrase.
- "predicate" is a lower snake_case relation such as works_at, lives_in, birth_year, favorite_dish or pet_name. \
Use the same predicate every time for the same relation.
- "statement" is one canonical third-person sentence that ends with the object, for example \
"Zorblat Quenn works at Vexmora Labs."
- "confidence" (0 to 1) is how firmly the text asserts the fact; hedged or second-hand facts get less. \
"importance" (0 to 1) is how useful the fact is to remember.
- "multi_valued" is true when the subject can hold several objects for this predicate at the same time \
(likes, speaks_language, has_child), otherwise false.
- "questions" holds up to 3 natural questions whose complete answer is the object; a question never contains \
the object.
- Extract only what the text states. Skip questions, requests, negated statements and small talk.
- In conversations, facts the user states about themselves use the subject "User" and statements such as \
"The user lives in Lisbon."; ignore facts that only the assistant asserted.
- Keep names and values in the language and spelling of the text.
If there are no facts, reply {"claims": []}."""

_QUESTIONS_PROMPT = """You write questions about a single fact. Reply with one JSON object and nothing else: \
{{"questions": ["...", "..."]}} with {n} different natural questions. Each question must be answerable from \
the fact alone, its complete answer must be the given answer, and it must not contain the answer. \
Vary the wording."""


def parse_json_payload(content: Any) -> Any:
    """Parse a model reply that should be JSON, tolerating ```json fences and text around the object."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty model reply")
    s = content.strip()
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
        end = max(s.rfind("}"), s.rfind("]"))
        if not starts or end <= min(starts):
            raise
        return json.loads(s[min(starts) : end + 1])


def parse_claims(payload: Any) -> list[ExtractedClaim]:
    """``{"claims": [...]}`` -> claims; malformed items are dropped, missing optional fields get defaults."""
    items = payload.get("claims") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError('model reply has no "claims" list')
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fields = {k: item.get(k) for k in ("subject", "predicate", "object")}
        if not all(isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip()
                   for v in fields.values()):
            continue
        subject = clean_value(str(fields["subject"]))
        if normalize(subject) in ("user", "the user", "i", "me"):
            subject = USER
        statement = item.get("statement")
        out.append(ExtractedClaim(
            subject=subject,
            predicate=canonical_predicate(str(fields["predicate"])),
            object=clean_value(str(fields["object"])),
            statement=statement if isinstance(statement, str) else "",
            confidence=_clamp(item.get("confidence"), 0.8),
            importance=_clamp(item.get("importance"), 0.5),
            multi_valued=_as_bool(item.get("multi_valued", False)),
            questions=_clean_questions(item.get("questions"), str(fields["object"]), 3),
        ))
    return prepare_claims(out)


def _chunks(text: str, max_chars: int) -> list[str]:
    """Split long text at paragraph boundaries into pieces of at most ``max_chars``."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        while len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            cut = para.rfind(". ", 0, max_chars) + 1 or max_chars
            chunks.append(para[:cut])
            para = para[cut:].lstrip()
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current)
    return chunks


class OpenAICompatExtractor:
    """Extractor backed by an OpenAI-compatible chat endpoint (DeepSeek by default).

    ``base_url`` defaults to ``$WIKIENGRAM_LLM_BASE_URL`` or the DeepSeek API; ``api_key`` to
    ``$WIKIENGRAM_LLM_API_KEY``, ``$DEEPSEEK_API_KEY`` or ``$OPENAI_API_KEY``. Requests are retried
    ``retries`` times on timeouts, connection errors, HTTP 5xx/429 and unparseable replies; other 4xx raise
    :class:`LLMError` at once.
    """

    retries = 2
    retry_backoff = 0.5  # seconds, doubled after each retry
    max_chars = 12_000  # longer texts are extracted chunk by chunk

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens  # servers that default to a short reply would cut the JSON off
        self.base_url = (base_url or os.environ.get("WIKIENGRAM_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = (
            api_key
            or os.environ.get("WIKIENGRAM_LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or None
        )
        self.timeout = timeout
        self.temperature = temperature
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __repr__(self) -> str:
        return f"OpenAICompatExtractor(model={self.model!r}, base_url={self.base_url!r})"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatExtractor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def extract(self, text: str, kind: str = "document") -> list[ExtractedClaim]:
        if not text.strip():
            return []
        hint = " The text contains only the user's messages." if kind == "conversation" else ""
        claims: list[ExtractedClaim] = []
        chunks = _chunks(text, self.max_chars)
        failed: list[LLMError] = []
        for chunk in chunks:
            user = f"Source kind: {kind}.{hint}\n\nText:\n{chunk}"
            try:
                claims.extend(self._chat(_EXTRACT_PROMPT, user, parse_claims))
            except LLMError as e:  # one bad reply must not lose the rest of a long document
                if e.status_code is not None and e.status_code < 500:
                    raise
                log.warning("extraction of one chunk (%d chars) failed and was skipped: %s", len(chunk), e)
                failed.append(e)
        if failed and len(failed) == len(chunks):
            raise failed[-1]
        return prepare_claims(claims)

    def paraphrase_questions(self, claim: Claim, n: int = 3) -> list[str]:
        if n <= 0:
            return []

        def parse(payload: Any) -> list[str]:
            questions = payload.get("questions") if isinstance(payload, dict) else payload
            if not isinstance(questions, list):
                raise ValueError('model reply has no "questions" list')
            return _clean_questions(questions, claim.object, n)

        user = f"Fact: {claim.statement}\nAnswer: {claim.object}"
        return self._chat(_QUESTIONS_PROMPT.format(n=n), user, parse)

    # ----------------------------------------------------------------------------------------- transport
    def _scrub(self, message: str) -> str:
        return message.replace(self._api_key, "***") if self._api_key else message

    def _chat(self, system: str, user: str, parse: Callable[[Any], T]) -> T:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        problem = "no attempt made"
        for attempt in range(self.retries + 1):
            if attempt:
                log.warning("LLM request failed (%s); retry %d/%d", problem, attempt, self.retries)
                time.sleep(self.retry_backoff * 2 ** (attempt - 1))
            try:
                response = self._client.post(url, json=body, headers=headers, timeout=self.timeout)
            except httpx.TimeoutException:
                problem = "timeout"
                continue
            except httpx.TransportError as e:
                problem = f"connection error: {type(e).__name__}"
                continue
            status = response.status_code
            if status >= 500 or status == 429:
                problem = f"HTTP {status}"
                continue
            if status >= 400:
                detail = " ".join(response.text.split())[:300]
                raise LLMError(self._scrub(f"LLM request to {url} failed with HTTP {status}: {detail}"), status)
            try:
                content = response.json()["choices"][0]["message"]["content"]
                return parse(parse_json_payload(content))
            except (ValueError, KeyError, IndexError, TypeError) as e:
                problem = f"unusable reply: {type(e).__name__}"
        raise LLMError(self._scrub(f"LLM request to {url} failed after {self.retries + 1} attempts ({problem})"))
