<!-- wikiengram:generated from .wikiengram/wiki.sqlite; edits here are overwritten, see SCHEMA.md -->
# strata_start

5 active claims · 4 historical versions

## Claims

- Strata_start sends the Strata server's output to .strata-mcp/server.log. — `c_24baeb94d4` v2 · confidence 1.00 · sources: [Strata docs/MCP_SERVER.md](../raw/s_ff6260a2da.md)
- Loading a model with strata_start takes 1-3 minutes. — `c_5eeb8a58e0` v2 · confidence 1.00 · sources: [Strata docs/MCP_SERVER.md](../raw/s_ff6260a2da.md)
- Strata_start polls GET /health until the model answers. — `c_84f62ea1a8` v2 · confidence 1.00 · sources: [Strata docs/MCP_SERVER.md](../raw/s_ff6260a2da.md)
- A model started with strata_start keeps running after the AI app closes, until strata_stop. — `c_c5681bd397` v2 · confidence 1.00 · sources: [Strata docs/MCP_SERVER.md](../raw/s_ff6260a2da.md)
- If a model is not ready after wait_seconds, 50 by default, strata_start returns loading. — `c_5f8ec71a41` v1 · confidence 0.95

## Related

- [[strata-stop]]

## History

- ~~Strata_start polls the health endpoint GET /health until the model answers.~~ — `c_84f62ea1a8` v1 · superseded · 2026-11-10 03:15 · ingest:s_ff6260a2da
- ~~Strata_start directs the server's output to .strata-mcp/server.log.~~ — `c_24baeb94d4` v1 · superseded · 2026-11-10 03:15 · ingest:s_ff6260a2da
- ~~Strata_start takes 1-3 minutes to load the model.~~ — `c_5eeb8a58e0` v1 · superseded · 2026-11-10 03:15 · ingest:s_ff6260a2da
- ~~Strata_start continues running even after the AI app closes.~~ — `c_c5681bd397` v1 · superseded · 2026-11-10 03:15 · ingest:s_ff6260a2da
