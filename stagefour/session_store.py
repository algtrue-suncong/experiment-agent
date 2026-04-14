"""
浏览器 Gateway 用的内存会话：每个 session_id 一份对话 messages，可多轮无限延续（仅受进程内存限制）。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ChatSession:
    id: str
    messages: List[dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    """简单的内存 Session 表；进程重启后丢失。后续可换 Redis / SQLite。"""

    def __init__(self, max_sessions: int = 5000):
        self._sessions: Dict[str, ChatSession] = {}
        self._meta_lock = asyncio.Lock()
        self._max_sessions = max_sessions

    async def get_or_create(self, session_id: Optional[str]) -> tuple[str, ChatSession]:
        async with self._meta_lock:
            sid = session_id or str(uuid.uuid4())
            if sid not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    raise RuntimeError("会话数已达上限，请稍后重试或清理旧会话")
                self._sessions[sid] = ChatSession(id=sid)
            sess = self._sessions[sid]
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
                out["first_session_messages"] = first.messages
            return out
