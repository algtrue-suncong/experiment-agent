import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TextContent:
    type: str = "text"
    text: str = ""


@dataclass
class ToolResult:
    content: List[TextContent]


class DidiStreamableHTTPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        base_url = os.getenv("DIDI_MCP_URL")
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=30.0
        )
        self.session_id = None

    async def initialize(self):
        resp = await self.client.post(
            f"mcp-servers?key={self.api_key}",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 0,
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "taxi-agent", "version": "1.0"}
                }
            }
        )
        data = resp.json()
        if "error" in data:
            raise Exception(f"初始化失败: {data['error']}")

        self.session_id = data["result"].get("sessionId")

        if self.session_id:
            print(f"  Session: {self.session_id[:8]}...")
        else:
            print(f"  Session: (无需session，使用URL Key认证)")

        await self.client.post(
            f"mcp-servers?key={self.api_key}",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"}
        )

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        resp = await self.client.post(
            f"mcp-servers?key={self.api_key}",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": 1,
                "params": {
                    "name": name,
                    "arguments": arguments
                }
            }
        )

        data = resp.json()

        if "result" in data and "content" in data["result"]:
            text = data["result"]["content"][0]["text"]
        else:
            text = json.dumps(data.get("result", data))

        return ToolResult(content=[TextContent(text=text)])

    async def list_tools(self):
        resp = await self.client.post(
            f"mcp-servers?key={self.api_key}",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2,
                "params": {}
            }
        )

        data = resp.json()

        tools = data.get("result", {}).get("tools", [])
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name=t["name"],
                description=t.get("description", ""),
                inputSchema=t.get("inputSchema", t.get("parameters", {}))
            ) for t in tools
        ])

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
