import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from didi_streamable_http_client import DidiStreamableHTTPClient

load_dotenv()

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class TaxiAgentPro:
    def __init__(self, interactive: bool = True):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.gaode_params = StdioServerParameters(
            command="python",
            args=[os.path.join(_STAGE_DIR, "gaode_mcp_server.py")],
            cwd=_STAGE_DIR,

            env={**os.environ, "GAODE_KEY": os.getenv("GAODE_KEY", "")},
        )

        self.test_mode = os.getenv("DIDI_TEST_MODE", "true").lower() == "true"
        self.user_phone = os.getenv("DIDI_PHONE", "")
        # 终端里可人工确认下单；Web/Gateway 场景下为 False，无法用 input()，非测试模式时危险操作会被拒绝
        self.interactive = interactive

        # 会话对象
        self.gaode_session: Optional[ClientSession] = None
        self.gaode_transport = None
        self.didi_session: Optional[DidiStreamableHTTPClient] = None

        self.all_tools = []
        self.current_order_id: Optional[str] = None

    async def initialize(self):
        # 初始化双MCP
        print("🚀 启动Pro版Agent" + ("（真实扣费模式）" if not self.test_mode else "（测试模式）"))

        # 使用mcp stdio client连接自定义高德MCP Server
        print("🔌 连接高德MCP...")
        self.gaode_transport = stdio_client(self.gaode_params)
        g_read, g_write = await self.gaode_transport.__aenter__()
        self.gaode_session = ClientSession(g_read, g_write)
        await self.gaode_session.__aenter__()
        await self.gaode_session.initialize()

        gaode_tools = await self.gaode_session.list_tools()
        print(f"✅ 高德MCP已连接，{len(gaode_tools.tools)}个工具")

        # 使用自定义滴滴MCP Client连接滴滴MCP Server（Streamable HTTP）
        print("🌐 连接滴滴MCP Pro...")
        self.didi_session = DidiStreamableHTTPClient(os.getenv("DIDI_MCP_KEY"))
        await self.didi_session.initialize()

        didi_tools = await self.didi_session.list_tools()
        print(f"✅ 滴滴MCP已连接，{len(didi_tools.tools)}个工具")
        print(f"   工具列表: {[t.name for t in didi_tools.tools]}")

        # 工具聚合
        self.all_tools = self._aggregate_tools(gaode_tools, didi_tools)
        print(f"🎯 总共可用 {len(self.all_tools)} 个工具")

    def _aggregate_tools(self, gaode_tools, didi_tools):
        # 工具聚合（加前缀区分来源）
        tools = []

        for t in gaode_tools.tools:
            tools.append({
                "server": "gaode",
                "name": f"gaode_{t.name}",
                "dangerous": False,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": f"gaode_{t.name}",
                        "description": f"[高德]{t.description}",
                        "parameters": t.inputSchema
                    }
                }
            })

        for t in didi_tools.tools:
            is_dangerous = t.name in ["taxi_create_order", "taxi_cancel_order"]
            prefix = "🚨[真实扣费]" if is_dangerous else "[滴滴]"

            tools.append({
                "server": "didi",
                "name": f"didi_{t.name}",
                "dangerous": is_dangerous,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": f"didi_{t.name}",
                        "description": f"{prefix}{t.description}",
                        "parameters": t.inputSchema
                    }
                }
            })
        return tools

    @staticmethod
    def _assistant_to_message_dict(msg: Any) -> Dict[str, Any]:
        if hasattr(msg, "model_dump"):
            d = msg.model_dump()
            return {k: v for k, v in d.items() if v is not None}
        return {
            "role": "assistant",
            "content": getattr(msg, "content", None),
            "tool_calls": getattr(msg, "tool_calls", None),
        }

    def _system_message(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": f"""你是智能出行助手（Pro版）。
{'当前为测试模式，不会真实扣费。' if self.test_mode else '⚠️当前为真实模式，下单会扣费！'}""",
        }

    async def execute_tool(self, tool_call):
        full_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        tool_info = next((t for t in self.all_tools if t["name"] == full_name), None)

        if tool_info and tool_info.get("dangerous"):
            if self.test_mode:
                return json.dumps({
                    "blocked": True,
                    "message": f"【测试模式拦截】{full_name}被禁止，避免真实扣费",
                    "args": args
                })

            if not self._confirm_order(args):
                return json.dumps({
                    "blocked": True,
                    "message": "用户未确认，取消操作（Web 场景请使用测试模式或后续接入显式确认流）"
                })

        if full_name.startswith("gaode_"):
            session = self.gaode_session
            real_name = full_name[6:]
        elif full_name.startswith("didi_"):
            session = self.didi_session
            real_name = full_name[5:]
        else:
            return json.dumps({"error": "未知工具"})

        print(f"🔧 [{tool_info['server'].upper()}] {real_name}({json.dumps(args, ensure_ascii=False)})")

        try:
            result = await session.call_tool(real_name, args)
            content = result.content[0].text
            return content
        except Exception as e:
            print(f"❌ 工具执行错误: {e}")
            return json.dumps({"error": str(e)})

    def _confirm_order(self, args):
        if not self.interactive:
            return False
        print("\n" + "=" * 50)
        print("🚨 即将创建真实订单（会扣费！）")
        print(f"起点: {args.get('from_name', '未知')}")
        print(f"终点: {args.get('to_name', '未知')}")
        print(f"车型: {args.get('product_category', 'economy')}")
        print(f"扣费账户: {self.user_phone}")
        print("=" * 50)

        confirm = input("确认下单吗？输入'确认'继续，其他取消: ")
        return confirm.strip() == "确认"

    async def _run_tool_loop(self, messages: List[dict]) -> str:
        openai_tools = [t["schema"] for t in self.all_tools]
        max_rounds = 5
        for round_num in range(max_rounds):
            if round_num >= max_rounds - 1:
                messages.append({
                    "role": "system",
                    "content": "已达到思考上限，请基于已获取的信息给出当前最佳答案，不要再调用工具。"
                })
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=messages,
                    tools=[],
                )
                final = response.choices[0].message
                text = (final.content or "").strip()
                messages.append({"role": "assistant", "content": text})
                return text

            resp = self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )

            msg = resp.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                messages.append(self._assistant_to_message_dict(msg))
                return text

            print(f"\n🔄 第{round_num + 1}轮")
            messages.append(self._assistant_to_message_dict(msg))

            for tc in msg.tool_calls:
                result = await self.execute_tool(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        return ""

    async def chat_turn(self, messages: List[dict], user_message: str) -> str:
        """
        在会话 messages 上追加一条用户消息并跑完一轮（含内部多轮工具调用）。
        会修改传入的 messages 列表（与 OpenAI 多轮格式一致）。
        """
        if not messages:
            messages.append(self._system_message())
        messages.append({"role": "user", "content": user_message})
        return await self._run_tool_loop(messages)

    async def run(self, query: str) -> str:
        messages: List[dict] = []
        return await self.chat_turn(messages, query)

    async def cleanup(self):
        print("\n🔌 清理资源...")

        if self.gaode_session:
            try:
                await self.gaode_session.__aexit__(None, None, None)
                print("高德会话已关闭")
            except Exception as e:
                print(f"高德会话清理出错: {e}")

        if self.gaode_transport:
            try:
                await self.gaode_transport.__aexit__(None, None, None)
                print("高德传输层已关闭")
            except Exception as e:
                print(f"高德传输层清理出错: {e}")

        if self.didi_session:
            try:
                await self.didi_session.__aexit__(None, None, None)
                print("滴滴会话已关闭")
            except Exception as e:
                print(f"滴滴会话清理出错: {e}")

        print("✅ 清理完成")


async def demo():
    agent = TaxiAgentPro()
    try:
        await agent.initialize()

        print("\n" + "=" * 60)
        print("🚗 智能出行助手已就绪（真实扣费模式）")
        print("输入 '退出' 结束对话")
        print("=" * 60)

        query = "帮我打车到成都市盛世锦都北门，我将在成都市领地环球金融中心A座南门上车，快车就行"

        print(f"\n👤 用户: {query}")
        result = await agent.run(query)
        print(f"🤖 Agent: {result}")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    # 环境检查
    required = ["DASHSCOPE_API_KEY", "GAODE_KEY", "DIDI_MCP_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"❌ 缺少环境变量: {missing}")
        sys.exit(1)

    asyncio.run(demo())
