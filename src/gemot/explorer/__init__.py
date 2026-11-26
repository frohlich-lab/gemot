"""Run-explorer: a checkout's model-building runs, in petab-mcp's viewer with their conversation.

The harness records a run in two disjoint stores joined by ``phoenix_trace_id``:

* **Phoenix** holds the conversation — the lead's prompts/responses, per-call token usage,
  the tool calls, and the analyst sub-agent trajectories (nested spans).
* the server's on-disk **version store** (``work/.petab_mcp/``) holds the model-variant
  trajectory — the Antimony + fit metrics (nllh/chi2) at each fit.

This package archives each run's on-disk half at run end so it survives the next ``cleanup()``
wipe (:mod:`.archive`), assembles the conversation from the trace (:mod:`.bundle`), and serves
each run in petab-mcp's viewer of its store with the conversation as a tab (:mod:`.app`,
:mod:`.conversation`). Entry point: ``python -m gemot.explorer`` (:mod:`.__main__`).
"""
