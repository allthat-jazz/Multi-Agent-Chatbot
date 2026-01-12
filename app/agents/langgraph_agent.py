import json

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from langchain_core.messages import SystemMessage, HumanMessage

from app.tools.kb_tools import build_kb_tools
from app.tools.db_tools import build_db_tools
from app.tools.web_tools import build_web_tools

# SYSTEM PROMPTS

PLANNER_SYSTEM = """Ты planner. Выбери ровно ОДНО: KB, DB или WEB.

KB: вопрос про документацию/ошибки/инструкции/как сделать (ищем в локальной базе знаний).
DB: вопрос про таблицы/SQL/данные в PostgreSQL (нужно выполнить запросы инструментами БД).
WEB: нужно "самое свежее" или в KB нет ответа (интернет-поиск).

Верни строго одно слово: KB или DB или WEB. Без пояснений.
"""

KB_SYSTEM = """Ты KB-агент (RAG). Используй ТОЛЬКО инструменты kb_*.

Правила:
- Сначала вызови kb_search по вопросу.
- Если hits пустые — прямо скажи "В KB нет данных" (без источников).
- НЕ пиши в ответе номера чанков, chunk_id, пути к файлам.
- В блоке "Источники" (если пишешь) указывай только уникальные ИМЕНА ФАЙЛОВ (например: runbook.md), без повторов, без путей.
"""

DB_SYSTEM = """Ты DB-агент. Используй ТОЛЬКО SQL инструменты (sql_db_*).

Правила:
- Если нужно получить данные — делай SELECT.
- Если нужно изменить данные — выполняй UPDATE/INSERT/DELETE и затем SELECT для проверки результата.
- Пиши итог + что было сделано.
"""

WEB_SYSTEM = """Ты WEB-агент. Используй ТОЛЬКО web_search.

Правила:
- Дай краткий ответ.
- В конце добавь 3-5 ссылок (URL) списком.
"""

class AgentState(BaseModel):
    messages: list = Field(default_factory=list)
    route: str = ""
    route2: str = ""
    kb_sources: list = Field(default_factory=list)

# HELPERS 

def _sget(state, key, default=None):
    if state is None:
        return default
    if hasattr(state, key):
        try:
            return getattr(state, key)
        except Exception:
            return default
    try:
        return state.get(key, default)
    except Exception:
        return default


def _last_user_text(messages):
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
        mt = getattr(m, "type", "")
        if mt in ("human", "user"):
            return (getattr(m, "content", "") or "").strip()
    return ""


def _fast_heuristic_route(q):
    low = (q or "").lower()
    if any(x in low for x in ["select ", "insert ", "update ", "delete ", "create table", "alter table", "postgres", "postgresql", "таблиц", "таблица", "sql", "schema", "pg_catalog"]):
        return "db"
    if any(x in low for x in ["свеже", "новост", "сегодня", "вчера", "последн", "internet", "гугл", "найди в интернете", "ссылк", "url"]):
        return "web"
    return "kb"


def _extract_kb_sources_from_messages(messages):
    srcs = []
    seen = set()

    for m in reversed(messages or []):
        name = getattr(m, "name", None) or ""
        tool_name = getattr(m, "tool", None) or getattr(m, "tool_name", None) or ""
        if name != "kb_search" and tool_name != "kb_search":
            continue
        content = getattr(m, "content", "") or ""
        try:
            data = json.loads(content)
        except Exception:
            data = None
        if isinstance(data, dict):
            hits = data.get("hits") or []
            for h in hits:
                s = (h.get("source") or "").strip()
                if not s:
                    continue
                s = s.replace("\\", "/").split("/")[-1]  # basename only
                if s in seen:
                    continue
                seen.add(s)
                srcs.append(s)
        break
    return srcs

def _kb_answer_says_no_data(text):
    low = (text or "").lower()
    return ("в kb нет данных" in low) or ("kb нет данных" in low) or ("нет данных в kb" in low)

# GRAPH

def build_langgraph(planner_llm, kb_agent_llm, db_agent_llm, web_agent_llm, rag, postgres_url):
    kb_tools = build_kb_tools(rag)
    db_tools = build_db_tools(db_agent_llm, postgres_url)
    web_tools = build_web_tools()

    kb_executor = create_react_agent(model=kb_agent_llm, tools=kb_tools)
    db_executor = create_react_agent(model=db_agent_llm, tools=db_tools)
    web_executor = create_react_agent(model=web_agent_llm, tools=web_tools)
    async def planner_node(state):
        q = _last_user_text(_sget(state, "messages", []))
        route = _fast_heuristic_route(q)
        if route == "kb":
            try:
                out = await planner_llm.ainvoke([SystemMessage(content=PLANNER_SYSTEM),
                    HumanMessage(content=q)])
                txt = (out.content or "").strip().upper()
                if "DB" in txt:
                    route = "db"
                elif "WEB" in txt:
                    route = "web"
                else:
                    route = "kb"
            except Exception:
                pass
        return {"route": route}

    async def kb_node(state):
        msgs = list(_sget(state, "messages", []) or [])
        msgs2 = msgs + [SystemMessage(content=KB_SYSTEM)]
        res = await kb_executor.ainvoke({"messages": msgs2})
        out_msgs = res.get("messages") or msgs2
        srcs = _extract_kb_sources_from_messages(out_msgs)
        last_text = ""
        if out_msgs:
            last_text = getattr(out_msgs[-1], "content", "") or ""
        route2 = "web" if (not srcs and _kb_answer_says_no_data(last_text)) else ""
        return {"messages": out_msgs, "kb_sources": srcs, "route2": route2}

    async def db_node(state):
        msgs = list(_sget(state, "messages", []) or [])
        msgs2 = msgs + [SystemMessage(content=DB_SYSTEM)]
        res = await db_executor.ainvoke({"messages": msgs2})
        out_msgs = res.get("messages") or msgs2
        return {"messages": out_msgs}

    async def web_node(state):
        msgs = list(_sget(state, "messages", []) or [])
        msgs2 = msgs + [SystemMessage(content=WEB_SYSTEM)]
        res = await web_executor.ainvoke({"messages": msgs2})
        out_msgs = res.get("messages") or msgs2
        return {"messages": out_msgs}

    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("kb", kb_node)
    g.add_node("db", db_node)
    g.add_node("web", web_node)

    g.set_entry_point("planner")

    def _route(state):
        return (_sget(state, "route", "") or "").strip()

    g.add_conditional_edges("planner", _route, {"kb": "kb", "db": "db", "web": "web"})

    def _route2(state):
        return (_sget(state, "route2", "") or "").strip()

    g.add_conditional_edges("kb", _route2, {"web": "web", "": END})
    g.add_edge("db", END)
    g.add_edge("web", END)

    return g.compile()
