"""
简易登录：从环境变量 GATEWAY_USERS 读取 user:pass 列表，签发随机 token（内存保存，进程重启失效）。
未配置 GATEWAY_USERS 时视为开发模式，不强制登录（用户记为 guest）。
"""
from __future__ import annotations

import os
import secrets
import time
from typing import Dict, Optional, Tuple


def _parse_users() -> Dict[str, str]:
    raw = os.getenv("GATEWAY_USERS", "").strip()
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        u, p = part.split(":", 1)
        u, p = u.strip(), p.strip()
        if u:
            out[u] = p
    return out


USERS = _parse_users()
# token -> (user_id, expiry_ts)
_TOKENS: Dict[str, Tuple[str, float]] = {}
_TOKEN_TTL_SEC = int(os.getenv("GATEWAY_TOKEN_TTL_SEC", str(60 * 60 * 24 * 30)))  # 默认 30 天


def auth_required() -> bool:
    return bool(USERS)


def login(username: str, password: str) -> Optional[str]:
    """校验用户名密码，成功返回 bearer token。"""
    if not USERS:
        return None
    if USERS.get(username) != password:
        return None
    token = secrets.token_urlsafe(32)
    _TOKENS[token] = (username, time.time() + _TOKEN_TTL_SEC)
    return token


def verify_token(token: Optional[str]) -> Optional[str]:
    """返回 user_id 或 None。"""
    if not token:
        return None
    row = _TOKENS.get(token)
    if not row:
        return None
    uid, exp = row
    if time.time() > exp:
        _TOKENS.pop(token, None)
        return None
    return uid


def resolve_user_id(authorization: Optional[str]) -> str:
    """
    从 Authorization: Bearer 解析用户；未配置 USERS 时返回 guest；
    已配置 USERS 且未提供有效 token 时返回空字符串表示未认证。
    """
    if not USERS:
        return "guest"
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""
    tok = authorization.split(None, 1)[1].strip()
    uid = verify_token(tok)
    return uid or ""
