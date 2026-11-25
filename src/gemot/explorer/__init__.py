"""Run-explorer: a slim web UI to inspect a single model-building run.

The harness records a run in two disjoint stores joined by ``phoenix_trace_id``:

* **Phoenix** holds the conversation — the lead's prompts/responses, per-call token usage,
  the tool calls, and the analyst sub-agent trajectories (nested spans).
* the server's on-disk **version store** (``work/.petab_mcp/``) holds the model-variant
  trajectory — the Antimony + fit metrics (nllh/chi2) at each fit.

This package assembles both into one view model (:mod:`.bundle`), archives each run's on-disk
half at run end so it survives the next ``cleanup()`` wipe (:mod:`.archive`), and serves a
scroll-synced viewer (:mod:`.server`). Entry point: ``python -m gemot.explorer`` (:mod:`.__main__`).
"""
