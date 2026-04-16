"""
浏览器 Gateway 用的内存会话：每 session_id 一份「短期记忆」结构（非完整 OpenAI messages 镜像）。
完整工具原文见 audit_store 落盘；模型侧仅使用最近 K 轮 + Session State + 滚动摘要。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatSession:
    id: str
    user_id: str = "guest"
    rolling_summary: str = ""
    session_state: Dict[str, Any] = field(default_factory=dict)
    # 每轮只保留最终可见的 user/assistant 文本（不含工具链明细）
    recent_exchanges: List[Dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    """简单的内存 Session 表；进程重启后丢失。后续可换 Redis / SQLite。"""

    def __init__(self, max_sessions: int = 5000):
        self._sessions: Dict[str, ChatSession] = {}
        self._meta_lock = asyncio.Lock()
        self._max_sessions = max_sessions

    async def get_or_create(self, session_id: Optional[str], user_id: str) -> tuple[str, ChatSession]:
        async with self._meta_lock:
            sid = session_id or str(uuid.uuid4())
            if sid not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    raise RuntimeError("会话数已达上限，请稍后重试或清理旧会话")
                self._sessions[sid] = ChatSession(id=sid, user_id=user_id)
                return sid, self._sessions[sid]

            sess = self._sessions[sid]
            # 登录用户与会话绑定：切换账号时丢弃旧 id，避免串数据
            if sess.user_id != user_id:
                sid = str(uuid.uuid4())
                self._sessions[sid] = ChatSession(id=sid, user_id=user_id)
                return sid, self._sessions[sid]

            sess.updated_at = time.time()
            return sid, sess

    async def delete(self, session_id: str) -> bool:
        async with self._meta_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def stats(self) -> dict:
        async with self._meta_lock:
            out: dict = {"sessions": len(self._sessions)}
            if self._sessions:
                first = next(iter(self._sessions.values()))
                out["first_session_recent"] = first.recent_exchanges[:2]
            return out
