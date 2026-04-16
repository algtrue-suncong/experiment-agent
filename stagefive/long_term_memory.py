"""
长期记忆：按用户维度使用 Markdown 文件存储（性格 + 稳定事实），支持关键词触发 + 轻量抽取写入。
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, List, Tuple

_MEMORY_LOCK = threading.Lock()

_STAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _STAGE_DIR / "data" / "memory"
MEMORY_DIR = Path(os.getenv("MEMORY_DATA_DIR", str(_DEFAULT_DATA)))

DEFAULT_MD_TEMPLATE = """## 性格
- 专业、简洁；优先用工具完成查价与叫车，少废话。

## 稳定事实
- （暂无）
"""

# 用户显式要求记住时的触发词（命中后再走抽取）
MEMORY_TRIGGER_PATTERN = re.compile(
    r"(记住|记得|别忘了|以后默认|默认|长期|一直|保存|存一下|帮我记下)",
    re.I,
)


def _safe_user_id(user_id: str) -> str:
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", user_id.strip() or "guest")
    return s[:120] if len(s) > 120 else s


def memory_path_for_user(user_id: str) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return MEMORY_DIR / f"{_safe_user_id(user_id)}.md"


def load_long_term_markdown(user_id: str) -> str:
    """读取用户 md；不存在则写入默认模板并返回。"""
    path = memory_path_for_user(user_id)
    with _MEMORY_LOCK:
        if not path.is_file():
            path.write_text(DEFAULT_MD_TEMPLATE, encoding="utf-8")
            return DEFAULT_MD_TEMPLATE
        return path.read_text(encoding="utf-8")


def _split_sections(md: str) -> Tuple[str, str]:
    """拆成 (性格块, 稳定事实块) 纯文本。"""
    text = md or ""
    parts = re.split(r"(?m)^##\s*", text)
    persona = ""
    facts = ""
    for i, p in enumerate(parts):
        if p.startswith("性格"):
            persona = p.split("\n", 1)[-1].strip()
        elif p.startswith("稳定事实") or p.startswith("事實"):
            facts = p.split("\n", 1)[-1].strip()
    if not persona and not facts:
        # 未按格式书写：整体当作事实区兜底
        return "", text.strip()
    return persona, facts


def _build_md(persona_lines: str, facts_lines: str) -> str:
    pl = (persona_lines or "").strip()
    fl = (facts_lines or "").strip()
    return f"""## 性格
{pl if pl else "- （未特别说明）"}

## 稳定事实
{fl if fl else "- （暂无）"}
"""


def append_stable_facts(user_id: str, new_bullets: List[str]) -> None:
    """在「稳定事实」下追加条目（去重简单按子串）。"""
    if not new_bullets:
        return
    md = load_long_term_markdown(user_id)
    persona, facts = _split_sections(md)
    existing = facts
    lines = [ln.strip() for ln in existing.splitlines() if ln.strip()]
    seen = set(lines)
    for b in new_bullets:
        b = b.strip()
        if not b:
            continue
        line = f"- {b}"
        if line in seen or b in seen:
            continue
        lines.append(line)
        seen.add(line)
    new_facts = "\n".join(lines)
    out = _build_md(persona, new_facts)
    path = memory_path_for_user(user_id)
    with _MEMORY_LOCK:
        path.write_text(out, encoding="utf-8")


def inject_long_term_into_system(base_system: str, user_id: str) -> str:
    """将 md 中两段注入 system（克制长度）。"""
    md = load_long_term_markdown(user_id)
    persona, facts = _split_sections(md)
    block = "【用户长期记忆（Markdown 维护：性格 + 稳定事实）】\n"
    if persona:
        block += "性格要点：\n" + persona[:1200] + "\n"
    if facts:
        block += "稳定事实：\n" + facts[:2000]
    merged = (base_system or "").rstrip() + "\n\n" + block.strip()
    return merged[:12000]  # 硬上限，避免撑爆 system


def extract_facts_with_llm(
    client: Any,
    *,
    user_message: str,
    assistant_reply: str,
    model: str = "qwen-plus",
) -> List[str]:
    """
    从一轮对话中抽取可长期保存的事实（列表）。
    仅在关键词触发后调用，控制成本。
    """
    prompt = f"""从下面用户与助手的一轮对话中，抽取适合「长期记住」的事实性条目（中文短句）。
规则：
- 只输出事实与偏好（例如常用上车点、默认车型、称呼），不要复述工具细节。
- 若没有值得长期保存的内容，输出空 JSON 数组。
- 输出严格为 JSON：{{"facts": ["...", "..."]}}

【用户】
{user_message}

【助手】
{assistant_reply}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        obj = json.loads(raw)
        facts = obj.get("facts") if isinstance(obj, dict) else None
        if isinstance(facts, list):
            return [str(x).strip() for x in facts if str(x).strip()]
    except Exception:
        pass
    # 兼容模型输出夹杂说明或 Markdown 代码块的情况
    m = re.search(r"\{[\s\S]*\"facts\"[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            facts = obj.get("facts") if isinstance(obj, dict) else None
            if isinstance(facts, list):
                return [str(x).strip() for x in facts if str(x).strip()]
        except Exception:
            pass
    return []


def maybe_persist_long_term(
    client: Any,
    *,
    user_id: str,
    user_message: str,
    assistant_reply: str,
) -> None:
    """关键词触发 + LLM 抽取后写入 md。"""
    if not (user_message or "").strip():
        return
    if not MEMORY_TRIGGER_PATTERN.search(user_message):
        return
    facts = extract_facts_with_llm(
        client, user_message=user_message, assistant_reply=assistant_reply or ""
    )
    append_stable_facts(user_id, facts)
