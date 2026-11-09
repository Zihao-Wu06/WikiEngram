# Contributing

Thanks for helping. A few rules keep the project trustworthy:

1. **No simulated results.** Benchmarks and examples must come from real training and generation. If a
   feature is not implemented or not validated on real hardware, the docs must say so.
2. **Engram parity is sacred.** Changes to `src/wikiengram/engram/{compression,layout,hashing,module}.py` must
   keep `tests/test_engram_parity.py` passing bit-exactly against DeepSeek's reference code.
3. **The backbone stays frozen.** Anything that trains must keep the byte-exact backbone and fusion checksums
   unchanged (the trainers raise otherwise).

## Development

```bash
pip install -e ".[dev,api,data]"
pytest -q            # unit, parity and end-to-end tests on a tiny random model (no downloads)
ruff check src tests scripts
```

Tests that load real pretrained models are marked `slow` and run with `WIKIENGRAM_SLOW=1`. Tests that
download files are marked `network`.

## Pull requests

* Keep PRs focused and include tests.
* For changes that affect memory quality, attach a `wikiengram bench` result (hardware, model, facts) before
  and after.
