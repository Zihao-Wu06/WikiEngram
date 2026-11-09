"""Insert benchmark result tables (results.md files from `wikiengram bench`) into README.md."""

from __future__ import annotations

import sys
from pathlib import Path

START, END = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"


def main(paths: list[str]) -> None:
    readme = Path("README.md")
    text = readme.read_text()
    body = "\n".join(Path(p).read_text().strip() + "\n" for p in paths)
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{body}{END}{tail}")


if __name__ == "__main__":
    main(sys.argv[1:])
