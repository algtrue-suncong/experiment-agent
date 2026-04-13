"""
本地 Gateway：HTTP API + 静态页，多会话并发；与进程内单个 TaxiAgentPro（共享 MCP）配合使用。

启动（在 stagefour 目录下）:
  pip install -r requirements.txt
  uvicorn gateway:app --host 0.0.0.0 --port 8765

浏览器打开: http://127.0.0.1:8765/
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

_STAGE_DIR = Path(__file__).resolve().parent
if str(_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE_DIR))

from didi_agent_pro import TaxiAgentPro  # noqa: E402
from session_store import SessionStore  # noqa: E402

session_store = SessionStore()
agent: TaxiAgentPro | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    missing = [k for k in ("DASHSCOPE_API_KEY", "GAODE_KEY", "DIDI_MCP_KEY") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"缺少环境变量: {missing}")

    agent = TaxiAgentPro(interactive=False)
    await agent.initialize()
    yield
    await agent.cleanup()
    agent = None


app = FastAPI(title="Didi Taxi Agent Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = _STAGE_DIR / "static"
if static_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


class ChatRequest(BaseModel):
    session_id: str | None = Field(None, description="首次可为空，服务端返回新 id；客户端需持久化")
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.get("/api/health")
async def health():
    st = await session_store.stats()
    return {"ok": True, "mcp": agent is not None, **st}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 未就绪")

    sid, sess = await session_store.get_or_create(body.session_id)
    async with sess.lock:
        reply = await agent.chat_turn(sess.messages, body.message.strip())
    return ChatResponse(session_id=sid, reply=reply)


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    ok = await session_store.delete(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@app.get("/")
async def index():
    index_path = static_dir / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="static/index.html 缺失")
    return FileResponse(index_path)


def main():
    import uvicorn

    host = os.getenv("GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("GATEWAY_PORT", "8765"))
    uvicorn.run("gateway:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
