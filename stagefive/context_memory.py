"""
短期上下文：工具结果压缩、会话状态启发式更新、滚动摘要与「最近 K 轮」裁剪。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

# 保留在模型侧展示给模型的最近「用户-助手」轮数（不含当前正在处理的一轮）
DEFAULT_RECENT_K = int(os.getenv("CONTEXT_RECENT_K", "6"))
# 单条工具结果压缩后的最大字符数
TOOL_COMPACT_MAX = int(os.getenv("CONTEXT_TOOL_COMPACT_MAX", "1200"))
# 滚动摘要最大长度（超出时截断旧摘要前缀，简单保护）
ROLLING_SUMMARY_MAX = int(os.getenv("CONTEXT_ROLLING_SUMMARY_MAX", "2500"))


def compress_tool_result(text: str, max_len: int = TOOL_COMPACT_MAX) -> str:
    """将工具原文压缩为较短 JSON/文本，供模型继续推理；审计仍使用原文。"""
    s = (text or "").strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            slim: Dict[str, Any] = {}
            priority_keys = (
                "order_id", "orderId", "oid", "status", "state",
                "message", "msg", "error", "code", "errno",
                "price", "fee", "distance", "duration", "eta",
                "route", "result", "data", "blocked",
            )
            for k in priority_keys:
                if k in obj and k not in slim:
                    v = obj[k]
                    if isinstance(v, (dict, list)):
                        slim[k] = json.dumps(v, ensure_ascii=False)[:400]
                    else:
                        slim[k] = v
            if not slim:
                for i, (k, v) in enumerate(list(obj.items())[:12]):
                    if isinstance(v, (dict, list)):
                        slim[k] = json.dumps(v, ensure_ascii=False)[:300]
                    else:
                        slim[k] = v
            out = json.dumps(slim, ensure_ascii=False)
            return out if len(out) <= max_len else out[: max_len - 3] + "..."
    except (json.JSONDecodeError, TypeError):
        pass
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _merge_rolling_summary(previous: str, addition: str, max_len: int = ROLLING_SUMMARY_MAX) -> str:
    """拼接摘要并做长度保护。"""
    prev = (previous or "").strip()
    add = (addition or "").strip()
    if not add:
        return prev
    merged = (prev + "\n" + add).strip() if prev else add
    if len(merged) <= max_len:
        return merged
    return merged[-max_len:]


def merge_session_state_heuristic(
    state: Dict[str, Any],
    *,
    user_message: str,
    assistant_reply: str,
    tool_full_texts: List[str],
) -> Dict[str, Any]:
    """
    用启发式合并会话状态（避免每轮额外 LLM 调用）。
    后续可替换为结构化抽取模型。
    """
    out = dict(state or {})
    text_blob = "\n".join(tool_full_texts[-8:])  # 只看最近几条工具原文做解析

    # 从 JSON 工具结果里抓 order_id
    for chunk in tool_full_texts:
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                for key in ("order_id", "orderId", "oid"):
                    if key in obj and obj[key]:
                        out["last_order_id"] = str(obj[key])
                        break
        except (json.JSONDecodeError, TypeError):
            continue

    # 简单地址线索（非常粗，仅作短期提示）
    place_pat = re.compile(r"(从|在|起点|上车)[:：]?\s*([^，。\n]{2,40})")
    m = place_pat.search(user_message or "")
    if m:
        out["from_hint"] = m.group(2).strip()
    place_pat2 = re.compile(r"(到|去|终点|下车)[:：]?\s*([^，。\n]{2,40})")
    m2 = place_pat2.search(user_message or "")
    if m2:
        out["to_hint"] = m2.group(2).strip()

    out["last_user_snippet"] = (user_message or "")[:200]
    out["last_assistant_snippet"] = (assistant_reply or "")[:200]
    # 保留最近工具名可由调用方写入；此处仅记录是否有工具痕迹
    out["had_tool_json"] = bool(text_blob.strip())
    return out


def summarize_old_exchange(
    client: Any,
    *,
    old_summary: str,
    user_text: str,
    assistant_text: str,
    model: str = "qwen-plus",
) -> str:
    """将一轮对话合并进滚动摘要（同步阻塞调用，适合在 chat 路径上低频触发）。"""
    prompt = f"""你是对话摘要助手。请将下面「旧摘要」与「一轮对话」压缩为简洁中文要点（条目前加 - ）。
保留：具体地址/POI、订单号、金额/车型偏好、用户明确约束。
丢弃：寒暄、重复、与出行无关的细节。

【旧摘要】
{old_summary or "（无）"}

【用户】
{user_text}

【助手】
{assistant_text}

直接输出摘要正文，不要标题。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    piece = (resp.choices[0].message.content or "").strip()
    return _merge_rolling_summary(old_summary, piece)


def trim_recent_exchanges(
    client: Any,
    recent: List[Dict[str, str]],
    rolling_summary: str,
    recent_k: int,
) -> Tuple[List[Dict[str, str]], str]:
    """
    保证 recent 长度不超过 recent_k：超出的最早几轮合并进 rolling_summary。
    recent 元素形如 {"user": "...", "assistant": "..."}
    """
    rs = rolling_summary
    buf = list(recent)
    while len(buf) > recent_k:
        first = buf.pop(0)
        rs = summarize_old_exchange(
            client,
            old_summary=rs,
            user_text=first.get("user", ""),
            assistant_text=first.get("assistant", ""),
        )
    return buf, rs


def format_context_system_blocks(
    *,
    rolling_summary: str,
    session_state: Dict[str, Any],
) -> str:
    """拼到 system 末尾的短上下文块。"""
    st = json.dumps(session_state or {}, ensure_ascii=False, indent=None)
    rs = (rolling_summary or "").strip()
    parts = []
    if rs:
        parts.append("【滚动摘要（较早轮次已压缩）】\n" + rs)
    parts.append("【会话状态（短期，可能不完整）】\n" + st)
    return "\n\n".join(parts)
