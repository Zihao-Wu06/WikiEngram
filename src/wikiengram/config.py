"""Workspace configuration (``wikiengram.toml``)."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class BackendConfig:
    kind: str = "hf"  # "hf": frozen Hugging Face decoder + attached V4.1-style Engram
    model: str = "Qwen/Qwen3-0.6B-Base"
    device: str = "auto"
    dtype: str = "auto"


@dataclass
class EngramConfig:
    layer_ids: list[int] = field(default_factory=lambda: [1, 10])
    max_ngram_size: int = 4
    n_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 65536  # prime search start per (n-gram order, head); rows per layer ~ 24 x this
    norm_eps: float = 1e-6


@dataclass
class AdaptConfigToml:
    corpus: str = "wikitext-2"
    max_tokens: int = 2_000_000
    steps: int = 300
    batch_size: int = 8
    seq_len: int = 128
    table_lr: float = 5e-3
    fusion_lr: float = 1e-3


@dataclass
class ConsolidationConfig:
    # selection policy
    min_score: float = 0.45
    min_confidence: float = 0.6
    stability_hours: float = 0.0  # a claim version must be this old before it is consolidated
    max_claims_per_run: int = 500
    w_importance: float = 0.4
    w_frequency: float = 0.3
    w_stability: float = 0.2
    w_confidence: float = 0.1
    frequency_saturation: int = 5
    stability_saturation_days: float = 7.0
    # training
    epochs: int = 20
    lr: float = 2e-2
    batch_size: int = 16
    questions_per_claim: int = 2
    include_statement: bool = True
    block_min_count: int = 5  # block n-grams seen >= N times in the background corpus (0 = off)
    block_shared_claims: int = 3  # block n-grams shared by samples of >= N different claims (0 = off)
    replay_overlapping: bool = True  # rehearse consolidated claims that genuinely share rows with new training
    min_replay_overlap: int = 4  # fewer shared rows (stray hash collisions) are frozen instead of rehearsed
    max_replay_claims: int = 200
    lazy_repair: bool = True  # after a deletion, retrain only affected claims whose recall probes now fail
    locality_weight: float = 0.0
    replay_corpus: str = "wikitext-2"
    keep_versions: int = 10


@dataclass
class FusionConfig:
    mode: str = "hybrid"  # none | wiki | engram | hybrid
    top_k: int = 5
    skip_verified_consolidated: bool = True  # hybrid: leave out context for claims Engram verifiably recalls
    delta_on_context: bool = False  # apply the personal delta to wiki context tokens too (default: question only)
    max_new_tokens: int = 32


@dataclass
class ExtractorConfig:
    kind: str = "rule"  # rule | openai
    model: str = "deepseek-chat"
    base_url: str = ""  # empty: WIKIENGRAM_LLM_BASE_URL or https://api.deepseek.com/v1 (key from env only)


@dataclass
class WorkspaceConfig:
    name: str = "default"
    backend: BackendConfig = field(default_factory=BackendConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)
    adapt: AdaptConfigToml = field(default_factory=AdaptConfigToml)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)

    @classmethod
    def load(cls, path: str | Path) -> WorkspaceConfig:
        return _from_dict(cls, tomllib.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(dump_toml(asdict(self)))


def _from_dict(cls, data: dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        kwargs[f.name] = _from_dict(type(default), value) if is_dataclass(default) else value
    return cls(**kwargs)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def dump_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for flat scalars plus one level of tables (all this config needs)."""
    lines = [f"{k} = {_toml_value(v)}" for k, v in data.items() if not isinstance(v, dict)]
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"\n[{k}]")
            lines.extend(f"{kk} = {_toml_value(vv)}" for kk, vv in v.items())
    return "\n".join(lines) + "\n"
