#!/usr/bin/env python3
"""
高德MCP Server (Python实现版)
功能完全对标官方：地理编码、路径规划
"""
import asyncio
import json
import os
import sys
import requests
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

load_dotenv()
app = Server("gaode-map")


def call_gaode(endpoint: str, params: dict) -> dict:
    """调用高德HTTP API"""
    key = os.getenv("GAODE_KEY")
    if not key:
        raise ValueError("缺少GAODE_KEY环境变量")

    url = f"https://restapi.amap.com/v3/{endpoint}"
    params["key"] = key

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "1":
            return {"error": data.get("info", "请求失败"), "code": data.get("infocode")}
        return data
    except Exception as e:
        return {"error": str(e)}


@app.call_tool()
async def handle_tool(name: str, arguments: dict):
    """处理工具调用"""

    if name == "geocode":
        addr = arguments.get("address")
        if not addr:
            return [TextContent(type="text", text=json.dumps({"error": "缺少address参数"}))]

        data = call_gaode("geocode/geo", {"address": addr})
        if "error" in data:
            return [TextContent(type="text", text=json.dumps(data))]

        codes = data.get("geocodes", [])
        if not codes:
            return [TextContent(type="text", text=json.dumps({"error": f"未找到地址: {addr}"}))]

        return [TextContent(type="text", text=json.dumps({
            "address": addr,
            "location": codes[0]["location"],  # "lng,lat"
            "province": codes[0].get("province"),
            "city": codes[0].get("city"),
            "formatted": codes[0].get("formatted_address")
        }, ensure_ascii=False))]

    elif name == "route_plan":
        origin = arguments.get("origin")
        dest = arguments.get("destination")
        strategy = arguments.get("strategy", 0)  # 0速度优先, 2距离优先, 3不走高速

        if not origin or not dest:
            return [TextContent(type="text", text=json.dumps({"error": "缺少起点或终点"}))]

        data = call_gaode("direction/driving", {
            "origin": origin,
            "destination": dest,
            "strategy": strategy,
            "extensions": "all"
        })

        if "error" in data:
            return [TextContent(type="text", text=json.dumps(data))]

        paths = data.get("route", {}).get("paths", [])
        if not paths:
            return [TextContent(type="text", text=json.dumps({"error": "无法规划路线"}))]

        path = paths[0]
        return [TextContent(type="text", text=json.dumps({
            "origin": origin,
            "destination": dest,
            "distance": int(path.get("distance", 0)),  # 米
            "duration": int(path.get("duration", 0)),  # 秒
            "tolls": int(path.get("tolls", 0)),  # 过路费元
            "strategy": strategy,
            "steps": [s.get("instruction", "") for s in path.get("steps", [])[:20]]  # 前2步提示
        }, ensure_ascii=False))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"未知工具: {name}"}))]


@app.list_tools()
async def list_tools():
    """注册工具"""
    return [
        Tool(
            name="geocode",
            description="将中文地址转换为经纬度坐标。支持省市区、街道、POI名称。",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "详细地址，如'北京市海淀区中关村'"}
                },
                "required": ["address"]
            }
        ),
        Tool(
            name="route_plan",
            description="规划驾车路线，返回距离(米)、时间(秒)、过路费、导航步骤。",
            inputSchema={
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "起点坐标'经度,纬度'"},
                    "destination": {"type": "string", "description": "终点坐标'经度,纬度'"},
                    "strategy": {"type": "integer", "description": "0-速度优先, 2-距离优先, 3-不走高速", "default": 0}
                },
                "required": ["origin", "destination"]
            }
        )
    ]


async def main():
    """启动Server"""
    if not os.getenv("GAODE_KEY"):
        print("❌ 错误: 请设置 GAODE_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    print("🚀 高德MCP Server (Python版) 启动中...", file=sys.stderr)
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
