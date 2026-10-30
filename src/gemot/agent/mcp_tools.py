"""The MCP -> LangChain tool bridge: every server tool the panel holds is built here.

LangChain's own ``langchain.mcp`` converts each tool and each result -- the successor of the
archived langchain-mcp-adapters, whose ``load_mcp_tools(session)`` this replaces, and which it
matches result for result (content blocks plus a ``structured_content`` artifact; an
``is_error`` result reaches the model as a failed ``ToolMessage``).

Its ``as_langchain_tool`` is written for a fastmcp ``Client``. It is handed
:class:`_SessionClient` instead, which calls through the mcp ``ClientSession`` the runner opened,
for one reason: the runner shadows that session's ``call_tool`` with the tool clock and the
per-call timeout (``runner.clock._wrap_tool_clock``, ``_wrap_call_timeout``), the bio cache
shadows it on a bio session (``bio_cache.wrap_session``), and a tool calling through a fastmcp
client of its own would bypass all three. What ``as_langchain_tool`` reads off its client is
pinned in tests/test_mcp_tools.py, since ``langchain.mcp`` is a beta namespace.
"""

from __future__ import annotations

import warnings
from typing import Any

#: Pages of ``list_tools`` read before giving up on a server whose cursor never ends.
_MAX_TOOL_PAGES = 100


class _SessionClient:
    """The part of a fastmcp ``Client`` that ``as_langchain_tool`` uses, over an mcp session.

    ``call_tool`` looks ``session.call_tool`` up on every call, so a shadow installed after the
    tools were built still applies.
    """

    #: Read into each tool's ``metadata["mcp"]["server"]``, which nothing here reads.
    server_info = None

    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> _SessionClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any], raise_on_error: bool = False):
        # `as_langchain_tool` asks for `raise_on_error=False`, which is all a ClientSession does:
        # an `is_error` result comes back as a result, not an exception.
        return await self._session.call_tool(name, arguments)


async def load_mcp_tools(session: Any) -> list:
    """Every tool ``session``'s server lists, as LangChain tools that call through ``session``."""
    from langchain_core._api import LangChainBetaWarning

    with warnings.catch_warnings():
        # Warned once per process on import; tests/test_mcp_tools.py is the answer to it.
        warnings.simplefilter("ignore", LangChainBetaWarning)
        from langchain.mcp import as_langchain_tool
    from mcp.types import PaginatedRequestParams

    client = _SessionClient(session)
    tools: list = []
    cursor = None
    for _ in range(_MAX_TOOL_PAGES):
        page = await session.list_tools(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        tools += [await as_langchain_tool(tool, client) for tool in page.tools]
        cursor = page.next_cursor
        if not cursor:
            return tools
    raise RuntimeError(f"list_tools returned a cursor on each of {_MAX_TOOL_PAGES} pages")
