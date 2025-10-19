"""Initiate a FastAPI application."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP
import logging 
from typing import Literal

from langchain_mcp_adapters.tools import to_fastmcp

from src.server.loader import load_yaml_config
from src.server.configuration import ServerConfiguration
from src.server.constants import PATH_TO_CONFIG
from src.server.app_context import AppContext, load_app_context
from src.server.sql_database.database import (
    query_sql_database_tool,
    info_sql_database_tool,
    list_sql_database_tool,
    query_sql_checker_tool,
)

# for auth implementation in servers, see example:
# https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/servers/simple-auth/mcp_simple_auth/server.py

logger = logging.getLogger(__name__)

   
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize on startup
    logger.info("Initializing app_lifespan...")
    try:
        yield load_app_context()
    finally:
        logger.info("Cleaning up app_lifespan...")

def create_mcp_server(config: ServerConfiguration) -> FastMCP:
    """Create a FastMCP server."""
    logger.info(f"Create app with port {config.port}, host {config.host}.")

    # source
    # https://langchain-ai.github.io/langgraph/tutorials/sql-agent/#3-customizing-the-agent
   
    langchain_tools = [
        list_sql_database_tool,
        info_sql_database_tool,
        query_sql_checker_tool,
        query_sql_database_tool,
    ]
    fastmcp_tools = [ 
        to_fastmcp(tool)
        for tool in langchain_tools 
    ]
    tool_names = ", ".join(
        [
            tool.name
            for tool in fastmcp_tools
        ]
    )
    logger.info(f"Create tools called {tool_names}")

    app = FastMCP(
        name="Customers",
        instructions="A MCP server to search Customers: name, age, email and countries.",
        lifespan=app_lifespan,
        host=config.host,
        port=config.port,
        debug=True,
        dependencies=[],
        tools=fastmcp_tools
    )

    return app


def load_server_config(port: int, host: str, transport: Literal["sse", "streamable-http"]) -> ServerConfiguration:
    """Run the MCP server with the specified port, host, and transport protocol."""
    
    config = load_yaml_config(PATH_TO_CONFIG)
    
    server_config = config.get("SERVER")
    if not server_config:
        raise ValueError(f"SERVER section is missing in {PATH_TO_CONFIG}.")
    settings = ServerConfiguration(**server_config)

    # take into account potential overrides
    if port:
        settings.port = port
    if host:
        settings.host = host
    if transport:
        settings.transport = transport

    return settings
