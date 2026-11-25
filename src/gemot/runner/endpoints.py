"""The bio-knowledge MCP endpoints the biocurator queries.

Its own module, holding nothing but these constants and importing nothing, so anything that needs
to know where the biocurator points — a CI guard checking the curated tool names still resolve
upstream, say — can import them without pulling in ``mcp`` and the rest of the runner.
"""

from __future__ import annotations

# Hardcoded bio-knowledge MCP endpoints the biocurator queries (no custom-endpoint config). Each
# is opened best-effort in its OWN supervised task by bio_sessions.open_bio_sessions: an
# unreachable server is skipped, a session that closes mid-run is reopened, and a transport failure
# at ANY point (connect OR live session) degrades the biocurator instead of aborting the run (issue
# #86). See the biocurator design in agent.py and the isolation rationale in bio_sessions.py.
_BIOCONTEXT_MCP_URL = (
    "https://biocontext-kb.fastmcp.app/mcp"  # BioContextAI Knowledgebase (hosted; fair-use)
)
_OMNIPATH_MCP_URL = "https://explore.omnipathdb.org/api/mcp"  # OmniPath MCP (saezlab/omnipath-next)
BIO_MCP_URLS = (_BIOCONTEXT_MCP_URL, _OMNIPATH_MCP_URL)
