"""Task 4 — MCP (Model Context Protocol) client utilities."""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def discover_tools(command: str, args: list[str] | None = None) -> list[dict]:
    """
    Connect to an MCP server over stdio and return its tool schemas.

    Usage:
        tools = asyncio.run(discover_tools("python", ["my_mcp_server.py"]))
    """
    server_params = StdioServerParameters(
        command=command,
        args=args or [],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tool_list = await session.list_tools()

            schemas = []
            for tool in tool_list.tools:
                schemas.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                })
            return schemas


async def call_mcp_tool(
    command: str,
    tool_name: str,
    arguments: dict,
    args: list[str] | None = None,
) -> str:
    """
    Call a specific tool on an MCP server and return the result.

    Usage:
        result = asyncio.run(call_mcp_tool("python", "web_search",
                                           {"query": "LangGraph"}, ["my_mcp_server.py"]))
    """
    server_params = StdioServerParameters(
        command=command,
        args=args or [],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return json.dumps([c.text for c in result.content])


# Quick test
if __name__ == "__main__":
    # Replace with your own MCP server command
    tools = asyncio.run(discover_tools("python", ["my_mcp_server.py"]))
    print(json.dumps(tools, indent=2))
