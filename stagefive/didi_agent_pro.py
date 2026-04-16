import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from audit_store import append_tool_audit, append_turn_audit
from context_memory import (
    DEFAULT_RECENT_K,
    compress_tool_result,
    format_context_system_blocks,
    merge_session_state_heuristic,
    trim_recent_exchanges,
)
from didi_streamable_http_client import DidiStreamableHTTPClient
from long_term_memory import inject_long_term_into_system, maybe_persist_long_term
from session_store import ChatSession

load_dotenv()

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class TaxiAgentPro:
    def __init__(self, interactive: bool = True):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
            tools.append(
                {
                    "server": "gaode",
                    "name": f"gaode_{t.name}",
                    "dangerous": False,
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": f"gaode_{t.name}",
                            "description": f"[高德]{t.description}",
                            "parameters": t.inputSchema,
                        },
                    },
                }
            )

        for t in didi_tools.tools:
            is_dangerous = t.name in ["taxi_create_order", "taxi_cancel_order"]
            prefix = "🚨[真实扣费]" if is_dangerous else "[滴滴]"

            tools.append(
                {
                    "server": "didi",
                    "name": f"didi_{t.name}",
                    "dangerous": is_dangerous,
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": f"didi_{t.name}",
                            "description": f"{prefix}{t.description}",
                            "parameters": t.inputSchema,
                        },
                    },
                }
            )
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

    def _base_system_text(self) -> str:
        return f"""你是智能出行助手（Pro版）。
{'当前为测试模式，不会真实扣费。' if self.test_mode else '⚠️当前为真实模式，下单会扣费！'}
回复尽量简洁；需要查价/路线/叫车时使用工具。"""

    def _build_system_content(self, sess: ChatSession) -> str:
        """system = 基础规则 + 长期记忆 + 滚动摘要 + 会话状态。"""
        assert isinstance(sess, ChatSession)
        base = self._base_system_text()
        base = inject_long_term_into_system(base, sess.user_id)
        blocks = format_context_system_blocks(
            rolling_summary=sess.rolling_summary,
            session_state=sess.session_state,
        )
        return base + "\n\n" + blocks

    async def execute_tool(self, tool_call, sess: ChatSession) -> Tuple[str, str]:
        """
        执行工具：返回 (压缩结果给模型, 原文用于审计与状态启发式)。
        """
        assert isinstance(sess, ChatSession)

        full_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")

        tool_info = next((t for t in self.all_tools if t["name"] == full_name), None)

        if tool_info and tool_info.get("dangerous"):
            if self.test_mode:
                raw = json.dumps(
                    {
                        "blocked": True,
                        "message": f"【测试模式拦截】{full_name}被禁止，避免真实扣费",
                        "args": args,
                    },
                    ensure_ascii=False,
                )
                compact = compress_tool_result(raw)
                append_tool_audit(
                    sess.id,
                    sess.user_id,
                    tool_name=full_name,
                    arguments=args,
                    result_full_text=raw,
                    compact_text=compact,
                    meta={"blocked": True},
                )
                return compact, raw

            if not self._confirm_order(args):
                raw = json.dumps(
                    {
                        "blocked": True,
                        "message": "用户未确认，取消操作（Web 场景请使用测试模式或后续接入显式确认流）",
                    },
                    ensure_ascii=False,
                )
                compact = compress_tool_result(raw)
                append_tool_audit(
                    sess.id,
                    sess.user_id,
                    tool_name=full_name,
                    arguments=args,
                    result_full_text=raw,
                    compact_text=compact,
                    meta={"blocked": True},
                )
                return compact, raw

        if full_name.startswith("gaode_"):
            session = self.gaode_session
            real_name = full_name[6:]
        elif full_name.startswith("didi_"):
            session = self.didi_session
            real_name = full_name[5:]
        else:
            err = json.dumps({"error": "未知工具"}, ensure_ascii=False)
            return compress_tool_result(err), err

        srv = (tool_info or {}).get("server", "unknown")
        print(f"🔧 [{str(srv).upper()}] {real_name}({json.dumps(args, ensure_ascii=False)})")

        try:
            result = await session.call_tool(real_name, args)
            content = result.content[0].text
            compact = compress_tool_result(content)
            append_tool_audit(
                sess.id,
                sess.user_id,
                tool_name=full_name,
                arguments=args,
                result_full_text=content,
                compact_text=compact,
                meta={"server": srv, "real_name": real_name},
            )
            return compact, content
        except Exception as e:
            print(f"❌ 工具执行错误: {e}")
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            compact = compress_tool_result(err)
            append_tool_audit(
                sess.id,
                sess.user_id,
                tool_name=full_name,
                arguments=args,
                result_full_text=err,
                compact_text=compact,
                meta={"error": True},
            )
            return compact, err

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

    async def _run_tool_loop(
        self,
        sess: ChatSession,
        messages: List[dict],
    ) -> Tuple[str, List[str]]:
        """
        多轮工具循环；messages 内含 system + 历史 + 当前用户。
        返回 (最终助手可见文本, 本轮工具原文列表)。
        """
        openai_tools = [t["schema"] for t in self.all_tools]
        max_rounds = 5
        tool_full_texts: List[str] = []

        for round_num in range(max_rounds):
            if round_num >= max_rounds - 1:
                messages.append(
                    {
                        "role": "system",
                        "content": "已达到思考上限，请基于已获取的信息给出当前最佳答案，不要再调用工具。",
                    }
                )
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=messages,
                    tools=[],
                )
                final = response.choices[0].message
                text = (final.content or "").strip()
                messages.append({"role": "assistant", "content": text})
                return text, tool_full_texts

            resp = self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            msg = resp.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                messages.append(self._assistant_to_message_dict(msg))
                return text.strip(), tool_full_texts

            print(f"\n🔄 第{round_num + 1}轮")
            messages.append(self._assistant_to_message_dict(msg))

            for tc in msg.tool_calls:
                compact, full_text = await self.execute_tool(tc, sess)
                tool_full_texts.append(full_text)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": compact,
                    }
                )

        return "", tool_full_texts

    async def chat_turn(self, sess: ChatSession, user_message: str) -> str:
        """
        使用「最近 K 轮 + Session State + 滚动摘要 + 长期记忆」构建上下文；
        工具原文写入审计，模型仅见压缩结果。
        """
        assert isinstance(sess, ChatSession)

        system_content = self._build_system_content(sess)
        messages: List[dict] = [{"role": "system", "content": system_content}]
        # 历史轮次（仅最终可见文本）
        for ex in sess.recent_exchanges:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": user_message})

        final_text, tool_full_texts = await self._run_tool_loop(sess, messages)

        # 短期会话状态（启发式）
        sess.session_state = merge_session_state_heuristic(
            sess.session_state,
            user_message=user_message,
            assistant_reply=final_text,
            tool_full_texts=tool_full_texts,
        )

        # 长期记忆：关键词触发时写入 md
        maybe_persist_long_term(
            self.client,
            user_id=sess.user_id,
            user_message=user_message,
            assistant_reply=final_text,
        )

        # 写入本轮到「最近 K 轮」，超出则滚动摘要
        sess.recent_exchanges.append({"user": user_message, "assistant": final_text})
        sess.recent_exchanges, sess.rolling_summary = trim_recent_exchanges(
            self.client,
            sess.recent_exchanges,
            sess.rolling_summary,
            DEFAULT_RECENT_K,
        )

        append_turn_audit(sess.id, sess.user_id, user_message=user_message, assistant_final=final_text)
        return final_text

    async def run(self, query: str) -> str:
        """CLI demo：单轮/多轮仍建议显式构造 ChatSession；此处用 guest 占位。"""
        demo_sess = ChatSession(id="cli-demo", user_id="guest")
        return await self.chat_turn(demo_sess, query)

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
