
from pydantic import BaseModel, Field
from typing import Literal, Literal


class ServerConfiguration(BaseModel):
    """Settings for the MCP server."""
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    transport: Literal["sse", "streamable-http"] = Field(default="streamable-http")

