"""Answer matching shared by verification, the benchmarks and the examples."""

from __future__ import annotations

import re


def _norm(s: str) -> str:
    # case, commas, periods and Markdown emphasis / code marks are formatting, not part of an answer
    return " ".join(s.lower().replace("**", "").replace("`", "").replace(".", " ").replace(",", " ").split())


def answer_in(prediction: str, answer: str) -> bool:
    """True if ``answer`` appears in ``prediction`` as whole words (case, commas and periods ignored), so a short
    answer such as "pi" does not match "api". A leading "the" in the answer is optional."""
    p, a = _norm(prediction), _norm(answer)
    if not p or not a:
        return False

    def has(x: str) -> bool:
        return re.search(r"(?<![a-z0-9])" + re.escape(x) + r"(?![a-z0-9])", p) is not None

    return has(a) or (a.startswith("the ") and has(a[4:]))
