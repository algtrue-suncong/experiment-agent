"""
工具与对话审计：工具原文落盘 jsonl，供服务端留存（模型侧仅使用压缩结果）。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()

_STAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_AUDIT = _STAGE_DIR / "data" / "audit"
AUDIT_DIR = Path(os.getenv("AUDIT_DATA_DIR", str(_DEFAULT_AUDIT)))


def _audit_path(session_id: str) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)[:200]
    return AUDIT_DIR / f"{safe}.jsonl"


def append_audit_record(
    session_id: str,
    user_id: str,
    record: Dict[str, Any],
) -> None:
    """追加一条审计记录（含工具原文等）。"""
    payload = {
        "ts": time.time(),
        "session_id": session_id,
        "user_id": user_id,
        **record,
    }
    path = _audit_path(session_id)
    line = json.dumps(payload, ensure_ascii=False)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def append_tool_audit(
    session_id: str,
    user_id: str,
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    result_full_text: str,
    compact_text: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    append_audit_record(
        session_id,
        user_id,
        {
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
            "result_full_text": result_full_text,
            "result_compact_text": compact_text,
            "meta": meta or {},
        },
    )


def append_turn_audit(
    session_id: str,
    user_id: str,
    *,
    user_message: str,
    assistant_final: str,
) -> None:
    """每轮结束再记一条用户可见轮次，便于对齐（不含工具原文，工具已单独记录）。"""
    append_audit_record(
        session_id,
        user_id,
        {
            "type": "turn",
            "user_message": user_message,
            "assistant_final": assistant_final,
        },
    )
