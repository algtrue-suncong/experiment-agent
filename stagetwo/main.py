#!/usr/bin/env python3
import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class TaxiAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # MCP Client 配置：连接你的 Python Server
        self.server_params = StdioServerParameters(
            command="python",
            args=["gaode_mcp_server.py"],
            env={"GAODE_KEY": os.getenv("GAODE_KEY")}
        )

    async def run(self, query: str):
        # 1. 启动 Server 子进程并建立连接
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 2. 获取工具列表（动态发现）
                tools_result = await session.list_tools()
                print(f"✅ 连接到 Server，发现 {len(tools_result.tools)} 个工具")

                # 3. 转换为 OpenAI 格式
                openai_tools = [{
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                } for t in tools_result.tools]

                # 4. 调用百炼模型决策
                messages = [
                    {"role": "system", "content": "你是出行助手，可用工具查询地图信息"},
                    {"role": "user", "content": query}
                ]

                max_rounds = 5
                for round_num in range(max_rounds):
                    if round_num >= max_rounds - 1:
                        # 强制让 LLM 基于现有信息给出最佳答案，不再允许调用工具
                        messages.append({
                            "role": "system",
                            "content": "已达到思考上限，请基于已获取的信息给出当前最佳答案，不要再调用工具。"
                        })
                        response = self.client.chat.completions.create(
                            model="qwen-plus",
                            messages=messages,
                            tools=[]  # 清空工具，强制直接回答
                        )
                        return response.choices[0].message.content

                    response = self.client.chat.completions.create(
                        model="qwen-plus",
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto"
                    )
                    msg = response.choices[0].message

                    # 如果不需要工具，任务完成
                    if not msg.tool_calls:
                        return msg.content

                    print(f"🔄 第{round_num + 1}轮：需要执行 {len(msg.tool_calls)} 个工具")

                    # 记录 LLM 的决策（我要调用这些工具）
                    messages.append(msg)

                    # 内层循环：并行/串行执行本轮所有工具调用
                    for tool_call in msg.tool_calls:
                        result = await session.call_tool(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments)
                        )
                        print(f"🔧 {tool_call.function.name} -> {result.content[0].text}")

                        # 把结果塞回对话历史
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.content[0].text
                        })

                    # 循环回去，让 LLM 基于新结果决定下一步


async def main():
    agent = TaxiAgent()
    result = await agent.run("从成都市领地环球金融中心A座南门（天府二街）到陕西省商南县政府怎么走？最短距离")
    print(f"🤖 Agent: {result}")


if __name__ == "__main__":
    asyncio.run(main())
