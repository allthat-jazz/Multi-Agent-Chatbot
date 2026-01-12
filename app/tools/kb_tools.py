import re
from langchain_core.tools import Tool

def build_kb_tools(rag):
    def _kb_search(query, k=5):
        hits = rag.search(query, k=int(k))
        out = []
        for h in hits:
            src = (h.get("source") or "").replace("\\", "/").split("/")[-1]
            out.append({"source": src, "text": h.get("text", ""), "score": float(h.get("score", 0.0)),})
        return {"hits": out}

    def _kb_reindex():
        rag.reindex()
        return {"ok": True}

    kb_search_tool = Tool(name="kb_search",
        description="Search in local KB. Input: query string. Returns JSON with hits (source,text,score).",
        func=lambda query: _kb_search(query, 5))

    def kb_search_k(query_and_k):
        q = query_and_k or ""
        m = re.search(r"\bk\s*=\s*(\d+)", q)
        k = int(m.group(1)) if m else 5
        q2 = re.sub(r"\bk\s*=\s*\d+\s*;?\s*", "", q).strip()
        if not q2:
            q2 = q.strip()
        return _kb_search(q2, k)

    kb_search_k_tool = Tool(name="kb_search_k",
        description="Search in local KB with custom k. Input: 'k=7; <your query>'. Returns JSON hits.",
        func=kb_search_k)

    kb_reindex_tool = Tool(name="kb_reindex",
        description="Reindex KB from local documents.",
        func=lambda _=None: _kb_reindex())

    return [kb_search_tool, kb_search_k_tool, kb_reindex_tool]
