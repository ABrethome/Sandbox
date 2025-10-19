"""Start MCP Server."""

import logging
from typing import Literal

import click

from src.server.app import create_mcp_server, load_server_config
from src.server.configuration import ServerConfiguration

logger = logging.getLogger(__name__)


@click.command()
@click.option("--port", default=None, help="Override port to listen on")
@click.option("--host", default=None, help="Override host to bind to")
@click.option(
    "--transport",
    default=None,
    type=click.Choice(["sse", "streamable-http"]),
    help="Override transport protocol to use ('sse' or 'streamable-http')",
)
def main(port: int, host: str, transport: Literal["sse", "streamable-http"]):
    """Run the simple MCP server."""
    logging.basicConfig(level=logging.INFO)
    
    server_config: ServerConfiguration = load_server_config(
        port=port,
        host=host,
        transport=transport,
    )

    mcp_server = create_mcp_server(server_config)

    logger.info(f"Starting server with {server_config.transport} transport")

    mcp_server.run(transport=server_config.transport)


if __name__ == "__main__":
    main()
