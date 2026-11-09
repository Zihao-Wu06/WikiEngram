"""Explicit memory (LLMWiki): extraction, reconciliation, versioning, retrieval, pages and persistence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from wikiengram.wiki import (
    ChangeEvent,
    Claim,
    ClaimStatus,
    EngramState,
    ExtractedClaim,
    Extractor,
    IngestReport,
    LintReport,
    LLMError,
    LLMWiki,
    OpenAICompatExtractor,
    RuleExtractor,
    Source,
)
from wikiengram.wiki.extract import parse_json_payload, split_sentences
from wikiengram.wiki.pages import WikiPages, slugify
from wikiengram.wiki.retrieval import BM25, ClaimIndex, tokenize

CLAIM_ID = re.compile(r"c_[0-9a-f]{10}")
LOG_ENTRY = re.compile(r"^## \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] (\w+) \| (.+)$", re.M)


@pytest.fixture
def wiki(tmp_path: Path):
    w = LLMWiki.init(tmp_path / "wiki")
    yield w
    w.close()


class StubExtractor:
    """Returns preset claims, so tests control confidence and multi-valuedness."""

    def __init__(self, *claims: ExtractedClaim) -> None:
        self.claims = list(claims)
        self.calls: list[tuple[str, str]] = []

    def extract(self, text: str, kind: str = "document") -> list[ExtractedClaim]:
        self.calls.append((text, kind))
        return list(self.claims)

    def paraphrase_questions(self, claim: Claim, n: int = 3) -> list[str]:
        return []


def works_at(obj: str, confidence: float = 0.8) -> ExtractedClaim:
    return ExtractedClaim("Zorblat Quenn", "works_at", obj, f"Zorblat Quenn works at {obj}.", confidence=confidence)


def section(page: str, title: str) -> str:
    """Body of a ``## title`` section of a generated page."""
    return page.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


# ------------------------------------------------------------------------------------------ layout & models
def test_init_creates_layout_and_is_idempotent(tmp_path):
    root = tmp_path / "wiki"
    w = LLMWiki.init(root)
    for rel in ("raw", "pages", ".wikiengram/wiki.sqlite", "index.md", "log.md", "SCHEMA.md"):
        assert (root / rel).exists(), rel
    assert w.stats()["claims"] == 0 and w.stats()["pages"] == 0
    w.close()
    (root / "SCHEMA.md").write_text("# my own conventions\n", encoding="utf-8")
    LLMWiki.init(root).close()
    assert (root / "SCHEMA.md").read_text(encoding="utf-8") == "# my own conventions\n"
    assert [op for op, _ in LOG_ENTRY.findall((root / "log.md").read_text(encoding="utf-8"))] == ["init"]
    with closing(sqlite3.connect(root / ".wikiengram" / "wiki.sqlite")) as conn:  # no WAL: root may be on SMB
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 1


def test_claim_roundtrip_and_key():
    c = Claim("c_0123456789", "  Zorblat   QUENN. ", "Works At", "Vexmora Labs", "Zorblat Quenn works at Vexmora Labs.",
              source_ids=["s_1"], questions=["Where does Zorblat Quenn work?"])
    assert c.key() == ("zorblat quenn", "works at")
    d = c.to_dict()
    assert d["status"] == "active" and d["engram_state"] == "none"
    assert json.loads(json.dumps(d)) == d
    assert Claim.from_dict({**d, "unknown_field": 1}) == c
    minimal = Claim.from_dict({"id": "c_1", "subject": "A", "predicate": "p", "object": "o", "statement": "A p o."})
    assert minimal.status is ClaimStatus.ACTIVE and minimal.engram_state is EngramState.NONE
    assert minimal.source_ids == [] and minimal.version == 1
    assert Claim("c_2", "A", "p", "o", "s", status="retracted").status is ClaimStatus.RETRACTED
    assert LintReport().ok() and not LintReport(low_confidence=["c_1"]).ok()
    assert IngestReport(None).added == []


# ----------------------------------------------------------------------------------------- rule extractor
BENCHMARK_FORMS = [
    ("Zorblat Quenn works at Vexmora Labs.", "Zorblat Quenn", "works_at", "Vexmora Labs",
     "Zorblat Quenn works at Vexmora Labs."),
    ("Zorblat Quenn is employed by Vexmora Labs.", "Zorblat Quenn", "works_at", "Vexmora Labs",
     "Zorblat Quenn works at Vexmora Labs."),
    ("The employer of Zorblat Quenn is Vexmora Labs.", "Zorblat Quenn", "works_at", "Vexmora Labs",
     "Zorblat Quenn works at Vexmora Labs."),
    ("Mirel Kesh lives in Port Elrin.", "Mirel Kesh", "lives_in", "Port Elrin", "Mirel Kesh lives in Port Elrin."),
    ("Mirel Kesh's home town is Port Elrin.", "Mirel Kesh", "lives_in", "Port Elrin",
     "Mirel Kesh lives in Port Elrin."),
    ("The city where Mirel Kesh lives is Port Elrin.", "Mirel Kesh", "lives_in", "Port Elrin",
     "Mirel Kesh lives in Port Elrin."),
    ("Tavik Orsin's dog is named Brix.", "Tavik Orsin", "pet_name", "Brix", "Tavik Orsin's dog is named Brix."),
    ("Tavik Orsin has a dog called Brix.", "Tavik Orsin", "pet_name", "Brix", "Tavik Orsin's dog is named Brix."),
    ("The name of Tavik Orsin's dog is Brix.", "Tavik Orsin", "pet_name", "Brix",
     "Tavik Orsin's dog is named Brix."),
    ("Kesh Varo's favorite dish is gromix stew.", "Kesh Varo", "favorite_dish", "gromix stew",
     "Kesh Varo's favorite dish is gromix stew."),
    ("Kesh Varo loves eating gromix stew.", "Kesh Varo", "favorite_dish", "gromix stew",
     "Kesh Varo's favorite dish is gromix stew."),
    ("The dish Kesh Varo likes best is gromix stew.", "Kesh Varo", "favorite_dish", "gromix stew",
     "Kesh Varo's favorite dish is gromix stew."),
    ("Orsin Tavik plays the drumla.", "Orsin Tavik", "instrument", "the drumla", "Orsin Tavik plays the drumla."),
    ("The instrument Orsin Tavik plays is the drumla.", "Orsin Tavik", "instrument", "the drumla",
     "Orsin Tavik plays the drumla."),
    ("Orsin Tavik is a player of the drumla.", "Orsin Tavik", "instrument", "the drumla",
     "Orsin Tavik plays the drumla."),
]


@pytest.mark.parametrize(("text", "subject", "predicate", "obj", "statement"), BENCHMARK_FORMS)
def test_rule_extractor_benchmark_relations(text, subject, predicate, obj, statement):
    [claim] = RuleExtractor().extract(text)
    assert (claim.subject, claim.predicate, claim.object, claim.statement) == (subject, predicate, obj, statement)
    assert claim.confidence == 0.8 and not claim.multi_valued
    assert len(claim.questions) == 3 and all(q.endswith("?") and obj not in q for q in claim.questions)


def test_rule_extractor_generic_and_multi_valued_relations():
    ex = RuleExtractor()
    triples = lambda text, kind="document": [  # noqa: E731
        (c.subject, c.predicate, c.object, c.statement, c.multi_valued) for c in ex.extract(text, kind)
    ]
    assert triples("Zorblat Quenn's favorite color is teal.") == [
        ("Zorblat Quenn", "favorite_color", "teal", "Zorblat Quenn's favorite color is teal.", False)]
    assert triples("The capital of Vexmora is Drumla City.") == [
        ("Vexmora", "capital", "Drumla City", "The capital of Vexmora is Drumla City.", False)]
    assert triples("Mirel Kesh's mother's name is Ana.")[0][:3] == ("Mirel Kesh", "mother_name", "Ana")
    assert triples("Zorblat Quenn likes chess. Zorblat Quenn enjoys long walks.") == [
        ("Zorblat Quenn", "likes", "chess", "Zorblat Quenn likes chess.", True),
        ("Zorblat Quenn", "likes", "long walks", "Zorblat Quenn likes long walks.", True)]
    assert triples("Zorblat Quenn was born in 1990.") == [
        ("Zorblat Quenn", "birth_year", "1990", "Zorblat Quenn was born in 1990.", False)]
    assert triples("Mirel Kesh was born in Port Elrin.") == [
        ("Mirel Kesh", "birth_place", "Port Elrin", "Mirel Kesh was born in Port Elrin.", False)]
    assert [t[1:3] for t in triples("Tavik Orsin was born in East Varo in 1987.")] == [
        ("birth_place", "East Varo"), ("birth_year", "1987")]
    assert triples("Zorblat Quenn plays chess.")[0][1:3] == ("plays", "chess")  # a game, not an instrument
    assert triples("Zorblat Quenn plays a key role in the project.") == []


def test_rule_extractor_markdown_sections():
    text = (
        "# Zorblat Quenn\n"
        "Employer: Vexmora Labs\n"
        "- **Home town:** Port Elrin\n"
        "- Shoe size: 44\n"
        "- Meeting at 10:30\n"
        "## Contact\n"  # a generic section: the subject stays Zorblat Quenn
        "Phone: 555-0100\n"
        "Update: she plays the drumla.\n"
        "```\nCode: not a fact\n```\n"
        "# Mirel Kesh\n"
        "Favorite Dish: gromix stew\n"
        "Her dog is named Brix.\n"
    )
    got = [(c.subject, c.predicate, c.object, c.statement) for c in RuleExtractor().extract(text)]
    assert got == [
        ("Zorblat Quenn", "works_at", "Vexmora Labs", "Zorblat Quenn works at Vexmora Labs."),
        ("Zorblat Quenn", "lives_in", "Port Elrin", "Zorblat Quenn lives in Port Elrin."),
        ("Zorblat Quenn", "shoe_size", "44", "Zorblat Quenn's shoe size is 44."),
        ("Zorblat Quenn", "phone", "555-0100", "Zorblat Quenn's phone is 555-0100."),
        ("Zorblat Quenn", "instrument", "the drumla", "Zorblat Quenn plays the drumla."),
        ("Mirel Kesh", "favorite_dish", "gromix stew", "Mirel Kesh's favorite dish is gromix stew."),
        ("Mirel Kesh", "pet_name", "Brix", "Mirel Kesh's dog is named Brix."),
    ]


def test_rule_extractor_maps_first_person_to_user_in_conversations():
    text = "Hi there! My dentist is Dr. Okafor. I live in Port Elrin. I work at Vexmora Labs, which is great."
    got = [(c.subject, c.predicate, c.object, c.statement) for c in RuleExtractor().extract(text, "conversation")]
    assert got == [
        ("User", "dentist", "Dr. Okafor", "The user's dentist is Dr. Okafor."),
        ("User", "lives_in", "Port Elrin", "The user lives in Port Elrin."),
        ("User", "works_at", "Vexmora Labs", "The user works at Vexmora Labs."),
    ]
    more = RuleExtractor().extract("I have a dog named Brix. My sister lives in Lisbon. I also like chess.", "conversation")
    assert [(c.subject, c.predicate, c.object) for c in more] == [
        ("User", "pet_name", "Brix"), ("User's sister", "lives_in", "Lisbon"), ("User", "likes", "chess")]
    assert more[1].statement == "The user's sister lives in Lisbon."
    assert RuleExtractor().extract("I live in Port Elrin.") == []  # documents: no speaker to resolve


def test_rule_extractor_hedges_negations_questions_and_cleanup():
    ex = RuleExtractor()
    [hedged] = ex.extract("Zorblat Quenn probably works at Acme Corp.")
    assert (hedged.object, hedged.confidence) == ("Acme Corp", 0.5)
    [believed] = ex.extract("I think that Mirel Kesh lives in Port Elrin.")
    assert (believed.subject, believed.confidence) == ("Mirel Kesh", 0.5)
    assert ex.extract("Zorblat Quenn doesn't work at Acme.") == []
    assert ex.extract("Zorblat Quenn no longer lives in Port Elrin.") == []
    assert ex.extract("Where does Zorblat Quenn work?") == []
    assert ex.extract("The dog lives in the garden.") == []  # not a named entity
    [quoted] = ex.extract("Zorblat McQuenn works at 'VEXMORA labs'!")
    assert (quoted.object, quoted.statement) == ("VEXMORA labs", "Zorblat McQuenn works at VEXMORA labs.")
    [cut] = ex.extract("Mirel Kesh lives in Port Elrin, where she grew up.")
    assert cut.object == "Port Elrin"
    assert len(ex.extract("Zorblat Quenn works at Vexmora Labs. Zorblat Quenn works at Vexmora Labs.")) == 1


def test_split_sentences_keeps_abbreviations_and_numbers():
    text = "Dr. Okafor met J. R. Tolkien at 3.14 p.m. today. Then he left! Acme Inc. Mirel lives here"
    assert split_sentences(text) == [
        "Dr. Okafor met J. R. Tolkien at 3.14 p.m. today.", "Then he left!", "Acme Inc.", "Mirel lives here"]
    assert split_sentences("张伟住在北京。他喜欢茶。") == ["张伟住在北京。", "他喜欢茶。"]


def test_rule_paraphrase_questions():
    ex = RuleExtractor()
    claim = Claim("c_1", "Zorblat Quenn", "works_at", "Vexmora Labs", "Zorblat Quenn works at Vexmora Labs.")
    assert ex.paraphrase_questions(claim, n=2) == ["Where does Zorblat Quenn work?", "Which company employs Zorblat Quenn?"]
    generic = Claim("c_2", "Zorblat Quenn", "shoe_size", "44", "Zorblat Quenn's shoe size is 44.")
    assert ex.paraphrase_questions(generic)[0] == "What is Zorblat Quenn's shoe size?"
    user = Claim("c_3", "User", "lives_in", "Port Elrin", "The user lives in Port Elrin.")
    assert ex.paraphrase_questions(user, n=1) == ["Where does the user live?"]
    assert ex.paraphrase_questions(claim, n=0) == []
    assert isinstance(ex, Extractor)


# ---------------------------------------------------------------------------------- ingest & reconcile
def test_ingest_adds_claims_with_provenance(wiki):
    text = "Zorblat Quenn works at Vexmora Labs. Mirel Kesh lives in Port Elrin."
    report = wiki.ingest_text(text, title="team.md")
    assert len(report.added) == 2 and not (report.updated or report.unchanged or report.conflicts)
    source = wiki.get_source(report.source_id)
    assert isinstance(source, Source) and source.title == "team.md" and source.kind == "document"
    assert source.path == f"raw/{report.source_id}.md" and source.sha256 == hashlib.sha256(text.encode()).hexdigest()
    claims = wiki.claims()
    assert [c.id for c in claims] == report.added  # document order
    for c in claims:
        assert CLAIM_ID.fullmatch(c.id) and c.version == 1 and c.status is ClaimStatus.ACTIVE
        assert c.source_ids == [report.source_id] and c.reason == f"ingest:{report.source_id}"
        assert c.page == slugify(c.subject) and c.questions and c.created_at > 0
    assert [(e.kind, e.old_version, e.new_version) for e in wiki.events()] == [("added", None, 1)] * 2
    assert all(isinstance(e, ChangeEvent) for e in wiki.events())


def test_reingesting_a_known_fact_merges_sources(wiki):
    first = wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.", title="a")
    second = wiki.ingest_text("The employer of Zorblat Quenn is Vexmora Labs.", title="b")
    assert second.unchanged == first.added and not second.added and not second.updated
    claim = wiki.get(first.added[0])
    assert claim.source_ids == [first.source_id, second.source_id] and claim.version == 1
    again = wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.", title="a again")  # identical bytes
    assert again.source_id == first.source_id and again.unchanged == first.added
    assert len(wiki.sources()) == 2 and len(list((wiki.root / "raw").iterdir())) == 2
    assert [e.kind for e in wiki.events()] == ["added"]


def test_unchanged_claims_keep_the_highest_confidence(tmp_path):
    stub = StubExtractor(works_at("Vexmora Labs", 0.6))
    with LLMWiki(tmp_path, extractor=stub) as w:
        [cid] = w.ingest_text("one").added
        stub.claims = [works_at("Vexmora Labs", 0.95)]
        w.ingest_text("two")
        stub.claims = [works_at("vexmora labs.", 0.3)]  # same object after normalization
        assert w.ingest_text("three").unchanged == [cid]
        assert w.get(cid).confidence == 0.95 and len(w.get(cid).source_ids) == 3
    assert stub.calls[0] == ("one", "document")


def test_newer_contradiction_supersedes(wiki):
    first = wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.")
    cid = first.added[0]
    second = wiki.ingest_text("Zorblat Quenn works at Gromix Foods.", title="news")
    assert second.updated == [cid] and not second.added and not second.conflicts
    claim = wiki.get(cid)
    assert (claim.version, claim.object, claim.statement) == (2, "Gromix Foods", "Zorblat Quenn works at Gromix Foods.")
    assert claim.source_ids == [second.source_id] and claim.reason == f"ingest:{second.source_id}"
    history = wiki.history(cid)
    assert [(c.version, c.status, c.object) for c in history] == [
        (1, ClaimStatus.SUPERSEDED, "Vexmora Labs"), (2, ClaimStatus.ACTIVE, "Gromix Foods")]
    events = wiki.events()
    assert [e.kind for e in events] == ["added", "updated"]
    assert (events[-1].old_version, events[-1].new_version, events[-1].reason) == (1, 2, f"ingest:{second.source_id}")
    assert [c.id for c in wiki.claims()] == [cid]
    assert [(c.id, c.version) for c in wiki.claims(ClaimStatus.SUPERSEDED)] == [(cid, 1)]


def test_low_confidence_contradiction_is_reported_not_applied(tmp_path):
    with LLMWiki(tmp_path / "a") as w:
        [cid] = w.ingest_text("Zorblat Quenn works at Vexmora Labs.").added
        report = w.ingest_text("Zorblat Quenn probably works at Gromix Foods.")  # 0.5 < 0.8 * 0.8
        assert not report.updated and not report.added
        [conflict] = report.conflicts
        assert conflict["claim_id"] == cid and conflict["current"] == "Vexmora Labs"
        assert conflict["proposed"] == "Gromix Foods" and conflict["proposed_confidence"] == 0.5
        assert "confidence" in conflict["reason"] and conflict["source_id"] == report.source_id
        assert w.get(cid).version == 1 and w.get(cid).object == "Vexmora Labs"
        assert [e.kind for e in w.events()] == ["added"]
        assert "conflict " + cid in (w.root / "log.md").read_text(encoding="utf-8")

    with LLMWiki(tmp_path / "b", supersede_ratio=0.5) as lenient:
        [cid] = lenient.ingest_text("Zorblat Quenn works at Vexmora Labs.").added
        assert lenient.ingest_text("Zorblat Quenn probably works at Gromix Foods.").updated == [cid]


def test_reingesting_an_old_source_keeps_later_corrections_and_deletions(wiki):
    text = "Zorblat Quenn works at Vexmora Labs. Zorblat Quenn likes chess."
    works, likes = wiki.ingest_text(text).added
    wiki.correct(works, object="Gromix Foods")
    wiki.retract(likes)
    again = wiki.ingest_text(text)
    assert not again.added and not again.updated and len(again.conflicts) == 2
    assert wiki.get(works).object == "Gromix Foods" and wiki.get(likes).status is ClaimStatus.RETRACTED
    assert [c.id for c in wiki.claims()] == [works]


def test_multi_valued_predicates_accumulate(wiki):
    a = wiki.ingest_text("Zorblat Quenn likes chess.").added
    b = wiki.ingest_text("Zorblat Quenn likes hiking.").added
    assert len(a) == len(b) == 1 and a != b
    assert wiki.ingest_text("Zorblat Quenn enjoys chess.").unchanged == a
    assert {c.object for c in wiki.claims()} == {"chess", "hiking"}
    assert all(c.multi_valued for c in wiki.claims())
    assert wiki.lint().contradictions == []


def test_ingest_conversation_extracts_from_user_turns_only(wiki):
    messages = [
        {"role": "user", "content": "My dentist is Dr. Okafor."},
        {"role": "assistant", "content": "Noted. Mirel Kesh lives in Port Elrin."},
        {"role": "user", "content": [{"type": "text", "text": "I live in Port Elrin."}]},
    ]
    report = wiki.ingest_conversation(messages)
    assert [(c.subject, c.predicate, c.object) for c in wiki.claims()] == [
        ("User", "dentist", "Dr. Okafor"), ("User", "lives_in", "Port Elrin")]
    source = wiki.get_source(report.source_id)
    assert source.kind == "conversation" and source.path.endswith(".json")
    assert json.loads((wiki.root / source.path).read_text(encoding="utf-8")) == messages
    with pytest.raises(TypeError):
        wiki.ingest_conversation("not a list")  # type: ignore[arg-type]


def test_ingest_file_and_input_validation(wiki, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Mirel Kesh\nEmployer: Gromix Foods\n", encoding="utf-8")
    report = wiki.ingest_file(path)
    source = wiki.get_source(report.source_id)
    assert source.title == "notes.md" and source.path.endswith(".md")
    assert (wiki.root / source.path).read_bytes() == path.read_bytes()
    assert wiki.get(report.added[0]).statement == "Mirel Kesh works at Gromix Foods."
    binary = tmp_path / "photo.png"
    binary.write_bytes(b"\x89PNG\x00\x00")
    with pytest.raises(ValueError):
        wiki.ingest_file(binary)
    with pytest.raises(ValueError):
        wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.", kind="email")
    with pytest.raises(ValueError):
        wiki.ingest_text("   ")


def test_failed_extraction_writes_nothing(wiki):
    class Broken(StubExtractor):
        def extract(self, text, kind="document"):
            raise RuntimeError("LLM down")

    wiki.extractor = Broken()
    with pytest.raises(RuntimeError):
        wiki.ingest_text("Zorblat Quenn works at Vexmora Labs.")
    assert wiki.sources() == [] and wiki.claims(None) == [] and list((wiki.root / "raw").iterdir()) == []


# ------------------------------------------------------------------------------- versioning & deletion
def test_correct_creates_a_version_and_marks_consolidated_claims_stale(wiki):
    c = wiki.add_claim("Zorblat Quenn", "works_at", "Vexmora Labs")
    assert (c.statement, c.confidence, c.version, c.reason) == ("Zorblat Quenn works at Vexmora Labs.", 1.0, 1, "add_claim")
    wiki.set_engram_state(c.id, EngramState.CONSOLIDATED)
    assert wiki.get(c.id).engram_state is EngramState.CONSOLIDATED and wiki.get(c.id).engram_version == 1
    assert len(wiki.history(c.id)) == 1 and [e.kind for e in wiki.events()] == ["added"]

    v2 = wiki.correct(c.id, object="Gromix Foods")
    assert (v2.id, v2.version, v2.statement) == (c.id, 2, "Zorblat Quenn works at Gromix Foods.")
    assert (v2.engram_state, v2.engram_version, v2.reason) == (EngramState.STALE, 1, "user correction")
    v3 = wiki.correct(c.id, statement="Zorblat Quenn is employed by Gromix Foods", reason="wording")
    assert (v3.version, v3.statement, v3.object) == (3, "Zorblat Quenn is employed by Gromix Foods.", "Gromix Foods")
    assert v3.engram_state is EngramState.STALE  # Engram memory still holds v1
    assert [(h.version, h.status) for h in wiki.history(c.id)] == [
        (1, ClaimStatus.SUPERSEDED), (2, ClaimStatus.SUPERSEDED), (3, ClaimStatus.ACTIVE)]
    assert wiki.history(c.id)[0].engram_state is EngramState.CONSOLIDATED
    assert wiki.correct(c.id, object="Gromix Foods").version == 3  # nothing changed: no new version
    assert [(e.kind, e.old_version, e.new_version) for e in wiki.events()] == [
        ("added", None, 1), ("updated", 1, 2), ("updated", 2, 3)]
    assert wiki.lint().stale_engram == [c.id]

    plain = wiki.add_claim("Mirel Kesh", "lives_in", "Port Elrin", confidence=0.6)
    corrected = wiki.correct(plain.id, object="East Varo")
    assert corrected.engram_state is EngramState.NONE and corrected.confidence == 1.0
    with pytest.raises(ValueError):
        wiki.correct(plain.id)
    with pytest.raises(KeyError):
        wiki.correct("c_0000000000", object="x")


def test_retract_and_restore(wiki):
    c = wiki.add_claim("Zorblat Quenn", "likes", "chess", multi_valued=True)
    wiki.set_engram_state(c.id, EngramState.CONSOLIDATED)
    r = wiki.retract(c.id)
    assert (r.status, r.version, r.engram_state, r.reason) == (
        ClaimStatus.RETRACTED, 1, EngramState.STALE, "user deletion")
    assert wiki.claims() == [] and [x.id for x in wiki.claims(ClaimStatus.RETRACTED)] == [c.id]
    assert [x.id for x in wiki.claims(None)] == [c.id]
    assert [h.status for h in wiki.history(c.id)] == [ClaimStatus.RETRACTED]
    assert wiki.search("chess") == [] and wiki.search("chess", include_inactive=True)[0][0].id == c.id
    assert wiki.retract(c.id).version == 1 and len(wiki.events()) == 2  # idempotent
    with pytest.raises(ValueError):
        wiki.correct(c.id, object="go")

    restored = wiki.restore(c.id)
    assert (restored.version, restored.status, restored.object) == (2, ClaimStatus.ACTIVE, "chess")
    assert restored.engram_state is EngramState.STALE
    assert [h.status for h in wiki.history(c.id)] == [ClaimStatus.RETRACTED, ClaimStatus.ACTIVE]
    assert wiki.restore(c.id).version == 2  # already active: no-op
    wiki.correct(c.id, object="go")
    rolled_back = wiki.restore(c.id, version=1)
    assert (rolled_back.version, rolled_back.object) == (4, "chess")
    assert [h.status for h in wiki.history(c.id)] == [
        ClaimStatus.RETRACTED, ClaimStatus.SUPERSEDED, ClaimStatus.SUPERSEDED, ClaimStatus.ACTIVE]
    assert [(e.kind, e.old_version, e.new_version) for e in wiki.events()] == [
        ("added", None, 1), ("retracted", 1, None), ("restored", 1, 2), ("updated", 2, 3), ("restored", 3, 4)]
    with pytest.raises(KeyError):
        wiki.restore(c.id, version=99)


def test_purge_removes_every_trace_of_the_claim(wiki):
    c = wiki.add_claim("Zorblat Quenn", "works_at", "Vexmora Labs")
    keep = wiki.add_claim("Mirel Kesh", "lives_in", "Port Elrin")
    wiki.correct(c.id, object="Gromix Foods", reason="she told me she moved to Gromix Foods")
    assert (wiki.root / "pages" / "zorblat-quenn.md").exists()
    wiki.purge(c.id)
    assert wiki.get(c.id) is None and wiki.history(c.id) == [] and [x.id for x in wiki.claims(None)] == [keep.id]
    events = [e for e in wiki.events() if e.claim_id == c.id]
    assert [e.kind for e in events] == ["added", "updated", "purged"] and all(e.reason == "" for e in events)
    assert (events[-1].old_version, events[-1].new_version) == (2, None)
    log = (wiki.root / "log.md").read_text(encoding="utf-8")
    assert "Gromix" not in log and "Vexmora" not in log and "Port Elrin" in log
    assert f"purge | {c.id}" in log and f"[redacted: purged claim {c.id}]" in log
    assert not (wiki.root / "pages" / "zorblat-quenn.md").exists()
    assert "zorblat" not in (wiki.root / "index.md").read_text(encoding="utf-8")
    with pytest.raises(KeyError):
        wiki.purge(c.id)


def test_history_order_and_events_since_seq(wiki):
    c = wiki.add_claim("Zorblat Quenn", "works_at", "Vexmora Labs")
    wiki.correct(c.id, object="Gromix Foods")
    wiki.correct(c.id, object="Drumla Media")
    assert [h.version for h in wiki.history(c.id)] == [1, 2, 3]
    assert [h.object for h in wiki.history(c.id)] == ["Vexmora Labs", "Gromix Foods", "Drumla Media"]
    events = wiki.events()
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == 3
    assert wiki.events(since_seq=seqs[0]) == events[1:]
    assert wiki.events(since_seq=seqs[-1]) == []
    assert wiki.history("c_unknown000") == []


def test_set_engram_state_and_touch_are_bookkeeping_only(wiki):
    c = wiki.add_claim("Zorblat Quenn", "works_at", "Vexmora Labs")
    wiki.correct(c.id, object="Gromix Foods")
    wiki.set_engram_state(c.id, EngramState.CONSOLIDATED, version=1)  # trained on v1, but v2 is current
    assert (wiki.get(c.id).engram_state, wiki.get(c.id).engram_version) == (EngramState.STALE, 1)
    wiki.set_engram_state(c.id, EngramState.CONSOLIDATED)
    assert (wiki.get(c.id).engram_state, wiki.get(c.id).engram_version) == (EngramState.CONSOLIDATED, 2)
    wiki.set_engram_state(c.id, EngramState.NONE)
    assert (wiki.get(c.id).engram_state, wiki.get(c.id).engram_version) == (EngramState.NONE, None)
    wiki.touch([c.id, c.id, "c_unknown000"])
    wiki.touch(c.id)
    claim = wiki.get(c.id)
    assert claim.access_count == 2 and claim.last_accessed is not None
    assert claim.version == 2 and len(wiki.history(c.id)) == 2 and len(wiki.events()) == 2
    assert wiki.history(c.id)[-1].access_count == 2  # the current version row stays in sync


# ------------------------------------------------------------------------------------------------ search
def test_search_ranks_by_bm25_with_subject_boost(wiki):
    works = wiki.add_claim("Zorblat Quenn", "works_at", "Vexmora Labs")
    plays = wiki.add_claim("Zorblat Quenn", "instrument", "the drumla")
    other = wiki.add_claim("Mirel Kesh", "works_at", "Gromix Foods")
    wiki.add_claim("Mirel Kesh", "lives_in", "Port Elrin")
    hits = wiki.search("Where does Zorblat Quenn work?", k=3)
    assert [c.id for c, _ in hits] == [works.id, plays.id, other.id]
    assert hits[0][1] > hits[1][1] > hits[2][1] > 0
    assert wiki.search("Who works at Gromix Foods?")[0][0].id == other.id
    assert len(wiki.search("Zorblat Quenn", k=1)) == 1
    assert wiki.search("xylophone") == [] and wiki.search("   ") == [] and wiki.search("Zorblat", k=0) == []
    wiki.touch([works.id])
    assert wiki.search("Zorblat Quenn work")[0][0].access_count == 1  # results reflect current bookkeeping


def test_search_cjk_and_first_person(wiki):
    zhang = wiki.add_claim("张伟", "lives_in", "北京", "张伟住在北京。")
    li = wiki.add_claim("李娜", "works_at", "上海银行", "李娜在上海银行工作。")
    assert zhang.statement == "张伟住在北京。" and zhang.page == "张伟"
    assert wiki.search("张伟住在哪里")[0][0].id == zhang.id
    assert wiki.search("李娜在哪里工作")[0][0].id == li.id
    wiki.ingest_conversation([{"role": "user", "content": "My dentist is Dr. Okafor."}])
    assert wiki.search("Who is my dentist?")[0][0].subject == "User"


def test_retrieval_primitives():
    assert tokenize("Zorblat Quenn's works, living and LIVES") == ["zorblat", "quenn", "work", "liv", "liv"]
    assert tokenize("I") == ["user"]
    cjk = tokenize("张伟住在北京")
    assert {"张", "伟", "张伟", "北京"} <= set(cjk)
    assert ClaimIndex([]).search("anything") == []
    assert BM25([]).scores(["x"]) == {}
    bm25 = BM25([("a", ["red", "apple"]), ("b", ["green", "apple", "apple"]), ("c", ["banana"])])
    scores = bm25.scores(["apple", "red"])
    assert set(scores) == {"a", "b"} and scores["a"] > scores["b"] > 0

    claims = [Claim("c_1", "Zorblat Quenn", "works_at", "Vexmora Labs", "Zorblat Quenn works at Vexmora Labs."),
              Claim("c_2", "Mirel Quenn", "works_at", "Gromix Foods", "Mirel Quenn works at Gromix Foods.")]
    plain = {c.id: s for c, s in ClaimIndex(claims, subject_boost=0.0).search("Where does Zorblat Quenn work?")}
    boosted = {c.id: s for c, s in ClaimIndex(claims).search("Where does Zorblat Quenn work?")}
    assert boosted["c_1"] == pytest.approx(1.5 * plain["c_1"])  # whole subject name in the query
    assert boosted["c_2"] == pytest.approx(plain["c_2"])  # only "Quenn" matches: no entity boost


def test_search_on_empty_wiki(wiki):
    assert wiki.search("anything") == [] and wiki.search("anything", include_inactive=True) == []


# -------------------------------------------------------------------------------------------------- lint
def test_lint_reports_contradictions_duplicates_low_confidence_stale_and_orphans(wiki):
    assert wiki.lint().ok()
    a = wiki.add_claim("Zorblat Quenn", "works_at", "Vexmora Labs")
    b = wiki.add_claim("Zorblat Quenn", "works_at", "Gromix Foods")
    d1 = wiki.add_claim("Mirel Kesh", "lives_in", "Port Elrin")
    d2 = wiki.add_claim("mirel kesh", "Lives In", "Port Elrin.")
    wiki.add_claim("Mirel Kesh", "likes", "tea", multi_valued=True)
    wiki.add_claim("Mirel Kesh", "likes", "chess", multi_valued=True)
    low = wiki.add_claim("Tavik Orsin", "instrument", "the drumla", confidence=0.3)
    gone = wiki.add_claim("Kesh Varo", "likes", "tea", multi_valued=True)
    wiki.set_engram_state(gone.id, EngramState.CONSOLIDATED)
    wiki.retract(gone.id)
    (wiki.root / "pages" / "ghost.md").write_text("# Ghost\n", encoding="utf-8")

    report = wiki.lint()
    assert report.contradictions == [{"subject": "Zorblat Quenn", "predicate": "works_at",
                                      "claim_ids": [a.id, b.id], "objects": ["Vexmora Labs", "Gromix Foods"]}]
    assert [d["claim_ids"] for d in report.duplicates] == [[d1.id, d2.id]]
    assert report.low_confidence == [low.id]
    assert report.stale_engram == [gone.id]
    assert report.orphan_pages == ["ghost", "kesh-varo"]
    assert not report.ok()
    assert wiki.lint(low_confidence=0.2).low_confidence == []


# ------------------------------------------------------------------------------------------------- pages
def test_pages_index_and_log(wiki):
    report = wiki.ingest_text(
        "Zorblat Quenn works at Vexmora Labs. Zorblat Quenn likes chess. Zorblat Quenn plays the drumla.",
        title="profile.md",
    )
    works, likes, plays = report.added
    wiki.correct(works, object="Gromix Foods")
    wiki.retract(likes)
    wiki.ingest_text("Mirel Kesh likes Zorblat Quenn.")
    page = (wiki.root / "pages" / "zorblat-quenn.md").read_text(encoding="utf-8")
    assert page.splitlines()[1] == "# Zorblat Quenn"
    active = section(page, "Claims")
    assert f"Zorblat Quenn works at Gromix Foods. — `{works}` v2" in active and f"`{plays}` v1" in active
    assert "chess" not in active and "Vexmora" not in active
    assert f"(../raw/{report.source_id}.md)" in active
    history = section(page, "History")
    assert f"~~Zorblat Quenn works at Vexmora Labs.~~ — `{works}` v1 · superseded" in history
    assert f"~~Zorblat Quenn likes chess.~~ — `{likes}` v1 · retracted" in history
    assert "[[mirel-kesh]]" in section(page, "Related")

    index = (wiki.root / "index.md").read_text(encoding="utf-8")
    assert "- [[zorblat-quenn]] — Zorblat Quenn works at Gromix Foods. — 2 claims" in index
    assert "## M" in index and "## Z" in index and "2 pages · 3 active claims" in index
    entries = LOG_ENTRY.findall((wiki.root / "log.md").read_text(encoding="utf-8"))
    assert entries == [("init", "wiki"), ("ingest", "profile.md"), ("correct", works), ("retract", likes),
                       ("ingest", "note")]

    (wiki.root / "pages" / "notes-by-hand.md").write_text("# mine\n", encoding="utf-8")
    snapshot = {p.name: p.read_text(encoding="utf-8") for p in (wiki.root / "pages").glob("*.md")}
    for p in (wiki.root / "pages").glob("*.md"):
        if p.name != "notes-by-hand.md":
            p.unlink()
    wiki.rebuild_pages()
    assert {p.name: p.read_text(encoding="utf-8") for p in (wiki.root / "pages").glob("*.md")} == snapshot


def test_slugify():
    assert slugify("Zorblat Quenn") == "zorblat-quenn"
    assert slugify("Dr. Okafor") == "dr-okafor" == slugify("dr okafor")
    assert slugify("张伟") == "张伟"
    assert slugify("Con") == "con-entity"  # reserved file name on Windows shares
    assert slugify("!!!").startswith("entity-")
    assert len(slugify("word " * 40)) <= 80


def test_raw_sources_are_immutable_copies(wiki):
    text = "Zorblat Quenn works at Vexmora Labs."
    report = wiki.ingest_text(text)
    source = wiki.get_source(report.source_id)
    raw = wiki.root / source.path
    assert raw.read_bytes() == text.encode() and hashlib.sha256(raw.read_bytes()).hexdigest() == source.sha256
    before = raw.stat().st_mtime_ns
    wiki.ingest_text(text)
    wiki.correct(report.added[0], object="Gromix Foods")
    assert raw.read_bytes() == text.encode() and raw.stat().st_mtime_ns == before
    with pytest.raises(RuntimeError):
        WikiPages(wiki.root).write_raw(source.path, b"tampered")
    assert raw.read_bytes() == text.encode()


# ------------------------------------------------------------------------------- OpenAI-compatible LLM
API_KEY = "sk-test-secret-123"


def make_llm(handler, **kwargs) -> OpenAICompatExtractor:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatExtractor("deepseek-chat", base_url="https://llm.test/v1/", api_key=API_KEY, client=client,
                                **kwargs)
    llm.retry_backoff = 0.0
    return llm


def reply(content: str | None, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


CLAIMS_JSON = json.dumps({"claims": [
    {"subject": "Zorblat Quenn", "predicate": "works_at", "object": "Vexmora Labs",
     "statement": "Zorblat Quenn works at Vexmora Labs.", "confidence": 0.9, "importance": 0.7,
     "multi_valued": False, "questions": ["Where does Zorblat Quenn work?", "Does he work at Vexmora Labs?"]},
    {"subject": "Mirel Kesh", "predicate": "likes", "object": "tea"},
    {"subject": "user", "predicate": "Home Town", "object": "Port Elrin", "multi_valued": "false",
     "confidence": "high"},
    {"subject": "Broken", "predicate": "has"},
    "not a claim",
]})


def test_llm_extractor_parses_valid_json():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return reply(CLAIMS_JSON)

    claims = make_llm(handler).extract("some text", kind="conversation")
    [request] = requests
    assert request.method == "POST" and str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "deepseek-chat" and body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0 and [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "conversation" in body["messages"][1]["content"] and "some text" in body["messages"][1]["content"]
    assert [(c.subject, c.predicate, c.object) for c in claims] == [
        ("Zorblat Quenn", "works_at", "Vexmora Labs"), ("Mirel Kesh", "likes", "tea"), ("User", "lives_in", "Port Elrin")]
    first, second, third = claims
    assert (first.confidence, first.importance, first.questions) == (0.9, 0.7, ["Where does Zorblat Quenn work?"])
    assert (second.confidence, second.importance, second.multi_valued, second.questions) == (0.8, 0.5, False, [])
    assert second.statement == "Mirel Kesh likes tea."
    assert third.statement == "The user lives in Port Elrin." and third.confidence == 0.8 and not third.multi_valued


def test_llm_extractor_accepts_fenced_json_and_surrounding_text():
    llm = make_llm(lambda request: reply(f"Here you go:\n```json\n{CLAIMS_JSON}\n```\nAnything else?"))
    assert len(llm.extract("text")) == 3
    assert parse_json_payload('Sure! {"claims": []} Hope this helps.') == {"claims": []}


def test_llm_extractor_retries_server_errors_and_unusable_replies():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500, text="upstream exploded")
        if len(calls) == 2:
            return reply("this is not json")
        return reply(CLAIMS_JSON)

    assert len(make_llm(handler).extract("text")) == 3 and len(calls) == 3


def test_llm_extractor_gives_up_after_retries():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(LLMError, match="3 attempts") as info:
        make_llm(handler).extract("text")
    assert len(calls) == 3 and "timeout" in str(info.value)


def test_llm_extractor_client_errors_raise_at_once_without_leaking_the_key():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": f"Authentication failed for key {API_KEY}"}})

    llm = make_llm(handler)
    with pytest.raises(LLMError) as info:
        llm.extract("text")
    err = info.value
    assert len(calls) == 1 and err.status_code == 401 and "401" in str(err)
    for text in (str(err), repr(err), repr(llm), str(err.__cause__), str(err.__context__)):
        assert API_KEY not in text


def test_llm_extractor_retries_rate_limits():
    responses = iter([httpx.Response(429, text="slow down"), reply('{"claims": []}')])
    assert make_llm(lambda request: next(responses)).extract("text") == []


def test_llm_paraphrase_questions_and_chunking():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "questions" in body["messages"][0]["content"] and "claims" not in body["messages"][0]["content"]:
            return reply(json.dumps({"questions": [
                "Where does Zorblat Quenn work?", "Is Vexmora Labs the employer of Zorblat Quenn?",
                "Which company employs Zorblat Quenn", "Where does Zorblat Quenn work?"]}))
        return reply(CLAIMS_JSON)

    llm = make_llm(handler)
    claim = Claim("c_1", "Zorblat Quenn", "works_at", "Vexmora Labs", "Zorblat Quenn works at Vexmora Labs.")
    assert llm.paraphrase_questions(claim, n=3) == [
        "Where does Zorblat Quenn work?", "Which company employs Zorblat Quenn?"]
    assert "Vexmora Labs" in seen[-1]["messages"][1]["content"]
    llm.max_chars = 40
    claims = llm.extract("First paragraph about things.\n\nSecond paragraph about more.\n\nThird one here.")
    assert len(seen) > 2 and len(claims) == 3  # one request per chunk, duplicates merged
    assert isinstance(llm, Extractor)


def test_llm_extractor_reads_environment(monkeypatch):
    monkeypatch.setenv("WIKIENGRAM_LLM_BASE_URL", "https://env.test/v1/")
    monkeypatch.delenv("WIKIENGRAM_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    seen = []
    client = httpx.Client(transport=httpx.MockTransport(lambda r: seen.append(r) or reply('{"claims": []}')))
    llm = OpenAICompatExtractor("deepseek-chat", client=client)
    assert llm.base_url == "https://env.test/v1" and llm.extract("x") == []
    assert seen[0].headers["authorization"] == "Bearer sk-from-env" and "sk-from-env" not in repr(llm)
    for name in ("WIKIENGRAM_LLM_BASE_URL", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert OpenAICompatExtractor("m", client=client).base_url == "https://api.deepseek.com/v1"


def test_wiki_with_llm_extractor(tmp_path):
    with LLMWiki(tmp_path, extractor=make_llm(lambda request: reply(CLAIMS_JSON))) as w:
        report = w.ingest_text("anything")
        assert len(report.added) == 3
        claim = w.get(report.added[0])
    assert claim.confidence == 0.9 and claim.questions == ["Where does Zorblat Quenn work?"]


# ----------------------------------------------------------------------------- persistence & threading
def test_state_persists_across_reopening(tmp_path):
    root = tmp_path / "wiki"
    with LLMWiki(root) as w1:
        report = w1.ingest_text("Zorblat Quenn works at Vexmora Labs. Zorblat Quenn likes chess.")
        works, likes = report.added
        w1.correct(works, object="Gromix Foods")
        w1.retract(likes)
        w1.set_engram_state(works, EngramState.CONSOLIDATED)
        w1.touch([works])
        before = (w1.claims(None), w1.history(works), w1.events(), w1.sources(), w1.stats())
    with LLMWiki(root) as w2:
        assert (w2.claims(None), w2.history(works), w2.events(), w2.sources(), w2.stats()) == before
        assert w2.get(works).engram_state is EngramState.CONSOLIDATED and w2.get(works).access_count == 1
        assert w2.search("Where does Zorblat Quenn work?")[0][0].id == works
        last = w2.events()[-1].seq
        w2.ingest_text("Mirel Kesh lives in Port Elrin.")
        assert w2.events(since_seq=last)[0].kind == "added"
        assert [op for op, _ in LOG_ENTRY.findall((root / "log.md").read_text(encoding="utf-8"))].count("init") == 1


def test_concurrent_use_from_threads(wiki):
    errors: list[Exception] = []
    lock = threading.Lock()

    def work(i: int) -> None:
        try:
            c = wiki.add_claim(f"Person {i}", "works_at", f"Company {i}")
            wiki.search(f"Where does Person {i} work?")
            wiki.touch([c.id])
        except Exception as e:
            with lock:
                errors.append(e)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(12)))
    assert errors == [] and len(wiki.claims()) == 12
    assert all(c.access_count == 1 for c in wiki.claims())
    assert len({e.seq for e in wiki.events()}) == 12
