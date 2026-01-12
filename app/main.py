import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import settings
from app.schemas import (AskRequest, AskResponse,
    SessionInfo, CreateSessionResponse,
    ReindexResponse, UploadResponse)
from app.rag.rag import HybridRAG
from app.memory.redis_history import get_history
from app.memory.sessions import create_session, list_sessions, get_title, set_title
from app.memory.sessions import delete_session
from app.agents.langgraph_agent import build_langgraph
from app.agents.llm import build_llm_openrouter

app = FastAPI(title="Agent System")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

redis_client = Redis.from_url(settings.redis_url, decode_responses=False)

def _safe_filename(name):
    name = (name or "").strip().replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9а-яА-Я._-]+", "_", name)
    return name or "doc.md"

def _auto_title(question, max_len=60):
    q = re.sub(r"\s+", " ", question.strip())
    if len(q) <= max_len:
        return q
    return q[: max_len - 1] + "…"

def _build_rag():
    return HybridRAG(kb_dir=settings.kb_dir,
        index_dir=settings.kb_index_dir,
        emb_model=settings.kb_emb_model,
        chunk_max_chars=settings.kb_chunk_max_chars,
        chunk_overlap_chars=settings.kb_chunk_overlap_chars,
        hybrid_alpha=settings.kb_hybrid_alpha,
        candidates=settings.kb_candidates,
        use_rerank=settings.kb_use_rerank,
        rerank_model=settings.kb_rerank_model,
        rerank_topn=settings.kb_rerank_topn)


@app.on_event("startup")
async def _startup():
    app.state.rag = _build_rag()
    app.state.rag.load_if_exists()
    llm = build_llm_openrouter(api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        site_url=settings.openrouter_site_url,
        app_name=settings.openrouter_app_name)

    app.state.graph = build_langgraph(planner_llm=llm,
        kb_agent_llm=llm,
        db_agent_llm=llm,
        web_agent_llm=llm,
        rag=app.state.rag,
        postgres_url=settings.postgres_url)

@app.get("/health")
async def health():
    rag = app.state.rag
    rag.load_if_exists()
    return {"ok": True,
        "kb_dir": str(settings.kb_dir),
        "kb_index_dir": str(settings.kb_index_dir),
        "kb_chunks_loaded": len(getattr(rag, "_chunks", []) or []),
        "redis_url": settings.redis_url,
        "postgres_url": settings.postgres_url,
        "sql_allow_write": settings.sql_allow_write,
        "model": settings.openrouter_model}

@app.post("/reindex", response_model=ReindexResponse)
async def reindex():
    rag = app.state.rag
    out = rag.reindex()
    return ReindexResponse(ok=True, docs=out["docs"], chunks=out["chunks"])

@app.post("/kb/upload", response_model=UploadResponse)
async def kb_upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved = []
    for f in files:
        fn = _safe_filename(f.filename)
        ext = Path(fn).suffix.lower()
        if ext not in {".md", ".txt", ".pdf", ".docx", ".csv"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {fn}")

        data = await f.read()
        if len(data) > settings.kb_max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"File too large: {fn}")

        out_path = settings.kb_dir / fn
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        saved.append({"filename": fn, "bytes": len(data)})

    rag = app.state.rag
    out = rag.reindex()
    return UploadResponse(ok=True,
        saved=saved,
        reindex=ReindexResponse(ok=True, docs=out["docs"], chunks=out["chunks"]),)

@app.get("/sessions", response_model=list[SessionInfo])
async def sessions_list():
    items = list_sessions(redis_client, limit=100)
    return [SessionInfo(**x) for x in items]

@app.post("/sessions", response_model=CreateSessionResponse)
async def sessions_create():
    s = create_session(redis_client, title="New chat")
    return CreateSessionResponse(**s)

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def session_get(session_id: str):
    title = get_title(redis_client, session_id)
    return SessionInfo(session_id=session_id, title=title)

@app.delete("/sessions/{session_id}")
async def session_delete(session_id: str):
    history = get_history(settings.redis_url, session_id)
    try:
        history.clear()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History delete failed: {e}")

    try:
        delete_session(redis_client, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session meta delete failed: {e}")

    return {"ok": True, "session_id": session_id}

def _trim_messages(msgs, max_turns):
    if max_turns <= 0:
        return msgs
    keep = 2 * max_turns
    return msgs[-keep:] if len(msgs) > keep else msgs

@app.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str, limit: int = 200):
    history = get_history(settings.redis_url, session_id)
    try:
        msgs = history.messages or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History load failed: {e}")

    if limit and limit > 0 and len(msgs) > limit:
        msgs = msgs[-limit:]

    out = []
    for m in msgs:
        mt = getattr(m, "type", "")
        content = getattr(m, "content", "")
        if isinstance(m, HumanMessage) or mt in ("human", "user"):
            role = "user"
        elif isinstance(m, AIMessage) or mt in ("ai", "assistant"):
            role = "assistant"
        else:
            role = "system"
        if role == "system":
            continue

        out.append({"role": role, "content": content})
    return {"session_id": session_id, "messages": out}

@app.post("/sessions/{session_id}/ask", response_model=AskResponse)
async def ask(session_id: str, payload: AskRequest):
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")
    
    history = get_history(settings.redis_url, session_id)
    prior = _trim_messages(history.messages, settings.chat_max_turns)
    
    rag = app.state.rag
    capture = {"kb_hits": [], "rag": rag}
    graph = app.state.graph
    state_in = {"messages": prior + [HumanMessage(content=payload.question)]}
    result_state = await graph.ainvoke(state_in)

    msgs = result_state.get("messages") or []
    answer = msgs[-1].content if msgs else ""

    history.add_user_message(payload.question)
    history.add_ai_message(answer)

    cur_title = get_title(redis_client, session_id)
    if cur_title == "New chat":
        set_title(redis_client, session_id, _auto_title(payload.question))
        
    kb_hits = capture.get("kb_hits") or []
    tools_used = []
    if kb_hits:
        tools_used.append("kb_search")

    return AskResponse(session_id=session_id,
        answer=(answer or "").strip(),
        tools_used=tools_used,
        error=None)
