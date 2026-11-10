<!-- wikiengram:generated from .wikiengram/wiki.sqlite; edits here are overwritten, see SCHEMA.md -->
# strata_stop

3 active claims · 4 historical versions

## Claims

- Strata_stop first sends POST /unload to the Strata server. — `c_b8fc829e68` v2 · confidence 1.00 · sources: [Strata docs/MCP_SERVER.md](../raw/s_ff6260a2da.md)
- While a request is running, strata_stop refuses unless it is called with force: true. — `c_e2608acc17` v2 · confidence 1.00 · sources: [Strata docs/MCP_SERVER.md](../raw/s_ff6260a2da.md)
- On Linux, strata_stop ends the Strata server with SIGTERM. — `c_da1720d0c1` v3 · confidence 1.00 · sources: [Strata docs/MCP_SERVER.md](../raw/s_ff6260a2da.md)

## Related

- [[strata-start]]

## History

- ~~Strata_stop sends an unload request POST /unload to the server.~~ — `c_b8fc829e68` v1 · superseded · 2026-11-10 03:15 · ingest:s_ff6260a2da
- ~~Strata_stop ends the server with SIGTERM on Linux.~~ — `c_da1720d0c1` v1 · superseded · 2026-11-10 03:15 · ingest:s_ff6260a2da
- ~~Strata_stop ends the server by process end on Windows.~~ — `c_da1720d0c1` v2 · superseded · 2026-11-10 03:15 · ingest:s_ff6260a2da
- ~~Strata_stop refuses to run while a request is active unless force: true is passed.~~ — `c_e2608acc17` v1 · superseded · 2026-11-10 03:15 · ingest:s_ff6260a2da
